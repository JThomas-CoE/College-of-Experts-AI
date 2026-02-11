"""
V12.1 Adaptive Template Router (Template Adapter)

Uses DeepSeek R1 Distill to:
1. Match user queries to framework templates via semantic similarity
2. Adapt templates to better match specific queries
3. Validate adapted frameworks

Flow:
- Find best template via cosine similarity
- If similarity > 0.95: use template as-is
- If 0.75 < similarity <= 0.95: adapt template with DeepSeek
- If similarity <= 0.75: decompose from scratch (no seed template)
"""

import asyncio
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import yaml
import numpy as np

from src.template_validator import TemplateValidator, ValidationResult
from src.deepseek_parser import parse_deepseek_response, ParsedResponse
from src.framework_scheduler import TaskFramework, FrameworkSlot

logger = logging.getLogger(__name__)


@dataclass
class AdaptationResult:
    """Result of template adaptation."""
    framework: TaskFramework
    rationale: str  # From DeepSeek R1 thinking or explicit rationale
    adaptation_type: str  # "adapted", "seed_fallback", "single_slot_fallback", "decompose_from_scratch"
    attempts: int
    validation_result: ValidationResult
    debug_info: Dict[str, Any] = field(default_factory=dict)


class TemplateAdapter:
    """
    V12.1 Adaptive Template Router (Persona-Aware Version).
    
    Uses DeepSeek R1 Distill 7B to adapt seed templates to specific queries.
    Supports both:
    - Template ADAPTATION: Modify an existing template to better match a query
    - Template DECOMPOSITION: Create a new framework from scratch when no template matches well
    
    Key Features:
    - Full persona context: Capabilities, guidelines, and output constraints
    - Intelligent slot routing: Each slot assigned to best-matching persona
    - Persona-specific prompts: Generate tailored prompts for each expert
    """
    
    def __init__(
        self,
        executor: Any,
        validator: TemplateValidator,
        prompt_template_path: Optional[str] = None,
        config: Optional[Dict] = None,
        persona_index_path: Optional[str] = None
    ):
        """
        Initialize the template adapter with persona awareness.
        
        Args:
            executor: DeepSeek model executor (must have generate() or generate_async())
            validator: TemplateValidator instance
            prompt_template_path: Path to prompt YAML file
            config: Override configuration dict
            persona_index_path: Path to persona index (auto-detected if None)
        """
        self.executor = executor
        self.validator = validator
        
        # Configuration with defaults
        self.config = {
            "temperature": 0.3,
            "max_tokens": 8192,
            "thinking_mode": True,
            "max_retries": 2,
            "internal_validation_retries": 1,
            "similarity_threshold": 0.95,
            "decomposition_similarity_threshold": 0.75,
            "persona_context_enabled": True,
        }
        if config:
            self.config.update(config)
        
        # Load prompt templates
        self.prompt_template = None
        self.retry_prompt_template = None
        self.decompose_from_scratch_prompt_template = None
        self.slot_prompt_template = None
        self.routing_guidelines = None
        
        # Persona context storage
        self.personas = {}  # persona_id -> persona_config
        self._persona_context_text = None
        
        if prompt_template_path:
            self._load_prompt_templates(prompt_template_path)
        
        # Load persona index and scopes
        self._load_persona_context(persona_index_path)
        
        # Track attempts for debug info
        self._attempt_count = 0
    
    def _load_prompt_templates(self, path: str):
        """Load prompt templates from YAML file."""
        try:
            with open(path) as f:
                prompts = yaml.safe_load(f)
            
            # Main adaptation prompt
            if "prompt" in prompts:
                self.prompt_template = prompts["prompt"]
            
            # Retry prompt
            if "retry_prompt" in prompts:
                self.retry_prompt_template = prompts["retry_prompt"]
            
            # Decompose from scratch prompt (new!)
            if "decompose_from_scratch_prompt" in prompts:
                self.decompose_from_scratch_prompt_template = prompts["decompose_from_scratch_prompt"]
            
            # Update config from YAML
            if "config" in prompts:
                for key, value in prompts["config"].items():
                    if key in self.config:
                        self.config[key] = value
            
            logger.info(f"[TemplateAdapter] Loaded prompts from {path}")
            
        except Exception as e:
            logger.warning(f"[TemplateAdapter] Failed to load prompts from {path}: {e}")
    
    def _load_persona_context(self, persona_index_path: Optional[str] = None):
        """
        Load persona context from the prompt template file.
        
        The persona context includes:
        - display_name: Human-readable name
        - capabilities: List of what the persona can do
        - guidelines: Persona-specific instructions
        - output_constraints: Format requirements
        """
        # Try to load from prompt template if it has persona definitions
        if self.prompt_template and "personas:" in self.prompt_template:
            try:
                import yaml
                # Parse the YAML section for personas
                personas_section = self.prompt_template.split("personas:")[1]
                # Find the end of personas (next major section)
                for marker in ["\n# =", "\nprompt:", "\nslot_prompt"]:
                    if marker in personas_section:
                        personas_section = personas_section.split(marker)[0]
                
                personas_data = yaml.safe_load(personas_section)
                if personas_data:
                    self.personas = personas_data
                    logger.info(f"[TemplateAdapter] Loaded {len(self.personas)} personas from template")
                    return
            except Exception as e:
                logger.warning(f"[TemplateAdapter] Failed to parse personas from template: {e}")
        
        # Also try loading from the expert_scopes/index.json if available
        if not persona_index_path:
            possible_paths = [
                Path("config/expert_scopes/index.json"),
                Path("../config/expert_scopes/index.json"),
            ]
            for p in possible_paths:
                if p.exists():
                    persona_index_path = str(p)
                    break
        
        if persona_index_path and Path(persona_index_path).exists():
            try:
                import json
                with open(persona_index_path) as f:
                    index_data = json.load(f)
                
                # Build basic persona entries from index
                for persona_id, info in index_data.get("personas", {}).items():
                    self.personas[persona_id] = {
                        "display_name": info.get("display_name", persona_id),
                        "savant": info.get("savant", "unknown"),
                        "capabilities": [f"General {info.get('category', 'general')} tasks"],
                        "guidelines": "- Follow best practices for the domain",
                        "output_constraints": {}
                    }
                
                logger.info(f"[TemplateAdapter] Loaded {len(self.personas)} personas from index")
            except Exception as e:
                logger.warning(f"[TemplateAdapter] Failed to load persona index: {e}")
        
        # Build the persona context text for prompts
        self._build_persona_context_text()
    
    def _build_persona_context_text(self):
        """Build a formatted text representation of all personas for prompt inclusion."""
        if not self.personas:
            self._persona_context_text = "No persona information available."
            return
        
        lines = []
        for persona_id, persona_info in self.personas.items():
            lines.append(f"\n### {persona_id}")
            lines.append(f"Display Name: {persona_info.get('display_name', persona_id)}")
            lines.append(f"Savant: {persona_info.get('savant', 'unknown')}")
            
            # Capabilities
            caps = persona_info.get('capabilities', [])
            if caps:
                lines.append("Capabilities:")
                for cap in caps[:5]:  # Limit to top 5
                    lines.append(f"  - {cap}")
            
            # Guidelines
            guidelines = persona_info.get('guidelines', '')
            if guidelines:
                lines.append("Guidelines:")
                for line in guidelines.strip().split('\n')[:5]:
                    if line.strip():
                        lines.append(f"  {line}")
            
            # Output constraints key hints
            constraints = persona_info.get('output_constraints', {})
            if constraints:
                const_lines = []
                for k, v in constraints.items():
                    if isinstance(v, bool):
                        if v:
                            const_lines.append(k)
                    else:
                        const_lines.append(f"{k}: {v}")
                if const_lines:
                    lines.append(f"Output: {', '.join(const_lines)}")
        
        self._persona_context_text = '\n'.join(lines)
        logger.debug(f"[TemplateAdapter] Built persona context ({len(self._persona_context_text)} chars)")
    
    def get_persona_context_for_ids(self, persona_ids: List[str]) -> str:
        """
        Get formatted persona context for a specific list of persona IDs.
        
        Args:
            persona_ids: List of persona IDs to include
            
        Returns:
            Formatted persona context string
        """
        if not self._persona_context_text:
            self._build_persona_context_text()
        
        if not persona_ids:
            return self._persona_context_text
        
        # Filter to only requested personas
        lines = self._persona_context_text.split('\n')
        filtered_lines = []
        include = True
        
        for line in lines:
            if line.startswith("### "):
                persona_id = line.replace("###", "").strip()
                include = persona_id in persona_ids
            if include:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines) if filtered_lines else self._persona_context_text
    
    def get_persona_info(self, persona_id: str) -> Optional[Dict]:
        """
        Get full information for a specific persona.
        
        Args:
            persona_id: The persona ID to look up
            
        Returns:
            Persona info dict or None if not found
        """
        return self.personas.get(persona_id)
    
    def generate_slot_prompt(
        self,
        slot_title: str,
        slot_description: str,
        expected_outputs: List[str],
        persona_id: str,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Generate a persona-specific prompt for executing a slot.
        
        Args:
            slot_title: Title of the slot
            slot_description: Description of what the slot produces
            expected_outputs: List of expected output types
            persona_id: The persona assigned to this slot
            additional_context: Any additional context from the framework
            
        Returns:
            Formatted prompt for the persona to execute
        """
        persona_info = self.get_persona_info(persona_id)
        
        if not persona_info:
            # Fallback to generic prompt
            return f"""## TASK
Execute slot: {slot_title}

Description: {slot_description}

Expected Outputs:\n{chr(10).join(f'- {o}' for o in expected_outputs)}

{additional_context or ''}"""
        
        # Build capabilities text
        capabilities = persona_info.get('capabilities', [])
        capabilities_text = "\n".join(f"  - {c}" for c in capabilities[:5])
        
        # Build guidelines text
        guidelines = persona_info.get('guidelines', '')
        guidelines_text = guidelines.strip() if guidelines else "No specific guidelines."
        
        # Build output constraints text
        constraints = persona_info.get('output_constraints', {})
        constraint_parts = []
        for k, v in constraints.items():
            if isinstance(v, bool):
                if v:
                    constraint_parts.append(k)
            else:
                constraint_parts.append(f"{k}: {v}")
        constraints_text = ", ".join(constraint_parts) if constraint_parts else "Standard format"
        
        # Format expected outputs
        outputs_text = "\n".join(f"- {o}" for o in expected_outputs)
        
        prompt = f"""## TASK
You are acting as {persona_info.get('display_name', persona_id)}.

Your expertise:
{capabilities_text}

Guidelines to follow:
{guidelines_text}

## SLOT CONTEXT
Slot: {slot_title}
Description: {slot_description}

Expected Outputs:
{outputs_text}
"""
        
        if additional_context:
            prompt += f"\n## ADDITIONAL CONTEXT\n{additional_context}\n"
        
        prompt += f"\n## OUTPUT REQUIREMENTS\n{constraints_text}\n"
        
        return prompt
    
    async def adapt(
        self,
        query: str,
        seed_template: Dict,
        personas: List[str],
        similarity_score: float = 0.5
    ) -> Optional[AdaptationResult]:
        """
        Adapt a seed template to better match a user query.
        
        Args:
            query: The user query to adapt to
            seed_template: The seed template to adapt
            personas: Available persona IDs for slot routing
            similarity_score: Similarity score of best matching template
            
        Returns:
            AdaptationResult with adapted framework, or None if failed
        """
        self._attempt_count += 1
        
        # Get timestamp for ID generation
        timestamp = int(time.time())
        original_id = seed_template.get("id", "unknown")
        
        # Build the adaptation prompt with persona context
        prompt = self._build_adaptation_prompt(
            query=query,
            seed_template=seed_template,
            personas_list=personas,
            timestamp=timestamp
        )
        
        # Add thinking mode prefix if configured
        if self.config.get("thinking_mode"):
            prompt = "<|im_start|>reasoning\n" + prompt
        
        logger.info(f"[TemplateAdapter] Adapting template '{original_id}' (similarity: {similarity_score:.3f})")
        
        # Build debug info structure
        debug_info = {
            "query": query,
            "seed_template_id": original_id,
            "best_template_similarity": float(similarity_score),
            "personas": personas,
            "attempts": []
        }
        
        try:
            # Main adaptation loop with retries
            for attempt in range(self.config.get("max_retries", 2)):
                attempt_info = {
                    "attempt": attempt + 1,
                    "timestamp": datetime.now().isoformat()
                }
                
                try:
                    # Generate with DeepSeek
                    logger.info(f"[TemplateAdapter] Attempt {attempt + 1}: Calling executor.generate()...")
                    response = await self._generate(prompt)
                    attempt_info["raw_response_length"] = len(response)
                    logger.info(f"[TemplateAdapter] Attempt {attempt + 1}: Got response ({len(response)} chars)")
                    logger.debug(f"[TemplateAdapter] Response preview: {response[:500]}...")
                    
                    # Parse response
                    parsed = parse_deepseek_response(response)
                    attempt_info["thinking_extracted"] = parsed.thinking is not None
                    attempt_info["rationale_extracted"] = parsed.rationale is not None
                    
                    if not parsed.success:
                        attempt_info["error"] = parsed.error
                        attempt_info["type"] = "parse_failure"
                        debug_info["attempts"].append(attempt_info)
                        
                        # Generate retry prompt
                        prompt = self._generate_retry_prompt(
                            base_prompt=prompt,
                            attempt=attempt,
                            error=parsed.error or "Parse failure"
                        )
                        continue
                    
                    # Validate adapted template
                    validation = self.validator.validate(parsed.json_data)
                    attempt_info["validation_passed"] = validation.is_valid
                    attempt_info["validation_errors"] = [str(e) for e in validation.errors]
                    attempt_info["validation_issues"] = validation.errors
                    attempt_info["validation_warnings"] = validation.warnings
                    
                    # INTERNAL VALIDATION RETRY LOOP
                    # If validation fails and we have retries left, auto-retry with fixed prompt
                    internal_retries = self.config.get("internal_validation_retries", 1)
                    current_validation = validation
                    current_json_data = parsed.json_data
                    current_attempt = 0
                    
                    while not current_validation.is_valid and current_attempt < internal_retries:
                        logger.info(f"Validation failed - auto-retry {current_attempt + 1}/{internal_retries}")
                        
                        # Step 1: LLM Review - get fresh context review of the failure
                        review_result = await self._review_decomposition_with_llm(
                            failed_json=current_json_data,
                            validation_errors=current_validation.errors,
                            query=query
                        )
                        
                        # Step 2: Generate fix prompt with reviewer guidance
                        fix_prompt = self._generate_validation_fix_prompt(
                            prompt,
                            current_validation.errors,
                            original_id,
                            timestamp,
                            failed_json=current_json_data,
                            review_result=review_result
                        )
                        
                        # Step 3: Fresh generation with fix guidance
                        fix_response = await self._generate(fix_prompt)
                        fix_parsed = parse_deepseek_response(fix_response)
                        
                        if fix_parsed.success:
                            current_json_data = fix_parsed.json_data
                            current_validation = self.validator.validate(current_json_data)
                            attempt_info["internal_retry_success"] = True
                        
                        current_attempt += 1
                    
                    # Final validation check
                    if not current_validation.is_valid:
                        attempt_info["type"] = "validation_failure"
                        attempt_info["validation_errors"] = [str(e) for e in current_validation.errors]
                        attempt_info["validation_issues"] = current_validation.errors
                        debug_info["attempts"].append(attempt_info)
                        
                        # Generate retry prompt
                        prompt = self._generate_retry_prompt(
                            base_prompt=prompt,
                            attempt=attempt,
                            error=f"Validation failed: {'; '.join(str(e) for e in current_validation.errors)}"
                        )
                        continue
                    
                    # Success!
                    attempt_info["type"] = "success"
                    attempt_info["validation_passed"] = True
                    debug_info["attempts"].append(attempt_info)
                    
                    framework = TaskFramework.from_dict(current_json_data)
                    
                    logger.info(f"[TemplateAdapter] Adaptation succeeded: {len(framework.slots)} slots")
                    
                    return AdaptationResult(
                        framework=framework,
                        rationale=parsed.rationale or parsed.thinking,
                        adaptation_type="adapted",
                        attempts=attempt + 1,
                        validation_result=current_validation,
                        debug_info=debug_info
                    )
                    
                except Exception as e:
                    attempt_info["error"] = str(e)
                    attempt_info["type"] = "exception"
                    debug_info["attempts"].append(attempt_info)
                    
                    # Generate retry prompt
                    prompt = self._generate_retry_prompt(
                        base_prompt=prompt,
                        attempt=attempt,
                        error=str(e)
                    )
                    continue
            
            # All retries exhausted - return seed template fallback
            logger.warning(f"[TemplateAdapter] All retries failed, using seed template")
            return self._create_seed_fallback(seed_template, timestamp, debug_info)
            
        except Exception as e:
            logger.exception(f"[TemplateAdapter] Adaptation failed: {e}")
            return self._create_seed_fallback(seed_template, timestamp, debug_info)
    
    def _build_adaptation_prompt(
        self,
        query: str,
        seed_template: Dict,
        personas_list: List[str],
        timestamp: int
    ) -> str:
        """Build the main adaptation prompt from template with persona context."""
        if self.prompt_template:
            # Use configured template with replace() to avoid curly brace conflicts
            prompt = self.prompt_template
            prompt = prompt.replace('{query}', query)
            prompt = prompt.replace('{seed_template}', json.dumps(seed_template, indent=2))
            prompt = prompt.replace('{timestamp}', str(timestamp))
            prompt = prompt.replace('{original_id}', seed_template.get("id", "unknown"))
            
            # Add persona context if enabled
            if self.config.get("persona_context_enabled", True):
                persona_context = self.get_persona_context_for_ids(personas_list)
                prompt = prompt.replace('{persona_context}', persona_context)
            else:
                # Fallback to simple list
                simple_list = "\n".join(f"- {p}" for p in personas_list) if personas_list else "- python_backend"
                prompt = prompt.replace('{persona_context}', simple_list)
            
            return prompt
        
        # Fallback: inline prompt
        original_id = seed_template.get("id", "unknown")
        persona_context = self.get_persona_context_for_ids(personas_list)
        return f"""<|im_start|>user
You are an expert task decomposition system. Adapt the seed template to match the query.

## USER QUERY
{query}

## SEED TEMPLATE
```json
{json.dumps(seed_template, indent=2)}
```

## AVAILABLE PERSONAS
{persona_context}

## TASK
Adapt this template to better match the query. You may:
- Keep slots that are relevant
- Remove slots that don't match
- Add new slots for missing functionality
- Modify dependencies

For each slot, specify the best persona_id based on their capabilities.

## OUTPUT FORMAT
RATIONALE: [Explain your changes]

ADAPTED FRAMEWORK:
```json
{{
    "id": "adapted_{original_id}_{timestamp}",
    "title": "[Title]",
    "description": "[Description]",
    "slots": [
        {{
            "id": "slot_id",
            "title": "Slot Title",
            "description": "What this slot produces",
            "persona": "[best_matching_persona_id]",
            "dependencies": [],
            "expected_outputs": ["output"]
        }}
    ]
}}
```
<|im_end|>
<|im_start|>assistant
"""
    
    def _generate_retry_prompt(self, base_prompt: str, attempt: int, error: str) -> str:
        """Generate a retry prompt with error context."""
        retry_header = f"""
================================================================================
RETRY ATTEMPT {attempt + 2}: Previous attempt had an issue
Error: {error}

Please fix this and output a corrected framework.
================================================================================

"""
        
        # Try to append to existing prompt
        if "<|im_start|>assistant" in base_prompt:
            # Insert after the assistant marker
            parts = base_prompt.split("<|im_start|>assistant", 1)
            return parts[0] + "<|im_start|>assistant\n" + retry_header + (parts[1] if len(parts) > 1 else "")
        else:
            return retry_header + base_prompt
    
    def _create_seed_fallback(
        self,
        seed_template: Dict,
        timestamp: int,
        debug_info: Dict
    ) -> AdaptationResult:
        """Create a fallback result using the seed template."""
        # Update template ID with timestamp
        original_id = seed_template.get("id", "unknown")
        seed_template["id"] = f"fallback_{original_id}_{timestamp}"
        
        framework = TaskFramework.from_dict(seed_template)
        validation = self.validator.validate(seed_template)
        
        debug_info["fallback_type"] = "seed_template"
        debug_info["rationale"] = "All adaptation attempts failed, using seed template"
        
        return AdaptationResult(
            framework=framework,
            rationale="Adapted template generation failed, using seed template",
            adaptation_type="seed_fallback",
            attempts=self._attempt_count,
            validation_result=validation,
            debug_info=debug_info
        )
    
    async def _review_decomposition_with_llm(
        self,
        failed_json: Dict,
        validation_errors: List,
        query: str
    ) -> Dict[str, Any]:
        """
        Use LLM as a reviewer to analyze decomposition failures and provide fix guidance.
        
        This creates a 'fresh context' review by using the reviewer persona.
        
        Returns:
            Dict with 'slot_intents' and 'fix_guidance'
        """
        # Load reviewer persona
        reviewer_path = Path("config/prompts/decomposition_reviewer.yaml")
        if not reviewer_path.exists():
            logger.warning("Decomposition reviewer persona not found, using fallback")
            return {
                "slot_intents": [],
                "fix_guidance": "Fix all validation errors"
            }
        
        try:
            with open(reviewer_path) as f:
                reviewer_config = yaml.safe_load(f)
            
            persona = reviewer_config.get("persona", "")
            output_format = reviewer_config.get("output_format", "")
            
            # Build reviewer prompt
            error_text = "\n".join(f"- {e}" for e in validation_errors)
            failed_json_str = json.dumps(failed_json, indent=2)
            
            review_prompt = f"""{persona}

## USER QUERY (Original Task)
{query}

## FRAMEWORK TO REVIEW
```json
{failed_json_str}
```

## VALIDATION ERRORS FOUND
{error_text}

## YOUR TASK
Review this framework and provide:
1. What each slot was INTENDED to do (preserve the logic)
2. Specific structural fixes needed

{output_format}
"""
            
            # Call LLM for review
            logger.info("[TemplateAdapter] Calling LLM reviewer for decomposition validation...")
            review_response = await self._generate(review_prompt)
            
            # Parse the review response
            # Extract slot intents and fix guidance
            slot_intents = []
            fix_guidance = []
            
            # Look for slot intent section
            if "SLOT_INTENT_SUMMARY:" in review_response or "Slot " in review_response:
                # Extract lines that look like slot descriptions
                for line in review_response.split("\n"):
                    if "was meant to:" in line or "persona:" in line:
                        slot_intents.append(line.strip())
            
            # Look for fix guidance
            if "FIX_GUIDANCE:" in review_response:
                guidance_section = review_response.split("FIX_GUIDANCE:")[1]
                if "DECISION:" in guidance_section:
                    guidance_section = guidance_section.split("DECISION:")[0]
                fix_guidance = [l.strip() for l in guidance_section.split("\n") if l.strip() and not l.strip().startswith("-")]
            
            return {
                "slot_intents": slot_intents,
                "fix_guidance": "\n".join(fix_guidance) if fix_guidance else "Fix structural errors",
                "raw_review": review_response
            }
            
        except Exception as e:
            logger.error(f"[TemplateAdapter] LLM review failed: {e}")
            return {
                "slot_intents": [],
                "fix_guidance": "Fix all validation errors",
                "raw_review": ""
            }
    
    def _generate_validation_fix_prompt(
        self,
        base_prompt: str,
        errors: List,
        original_id: str,
        timestamp: int,
        failed_json: Optional[Dict] = None,
        review_result: Optional[Dict] = None
    ) -> str:
        """Generate a prompt specifically designed to fix validation errors."""
        # Build error list
        error_lines = []
        for i, error in enumerate(errors, 1):
            error_lines.append(f"{i}. {error}")
        
        error_text = "\n".join(error_lines)
        
        # Extract slot summary from failed JSON (preserve descriptions!)
        slot_summary = ""
        if failed_json and "slots" in failed_json:
            slots = failed_json.get("slots", [])
            if slots:
                slot_summary = "INTENDED SLOTS (fix IDs/formatting ONLY - preserve descriptions):\n"
                for slot in slots:
                    slot_id = slot.get("id", "unknown")
                    slot_persona = slot.get("persona", "unknown")
                    slot_title = slot.get("title", "Unknown")
                    slot_desc = slot.get("description", "")
                    slot_summary += f'  - "{slot_id}" (persona: {slot_persona}) - {slot_title}\n'
                    if slot_desc:
                        slot_summary += f'    Description: {slot_desc}\n'
        
        # Include review guidance if available
        review_section = ""
        if review_result:
            if review_result.get("slot_intents"):
                review_section += "\nINTENDED SLOT PURPOSES (preserve these):\n"
                for intent in review_result["slot_intents"]:
                    review_section += f"  {intent}\n"
            if review_result.get("fix_guidance"):
                review_section += f"\nREVIEWER FIX GUIDANCE:\n{review_result['fix_guidance']}\n"
        
        fix_guidance = f"""

================================================================================
VALIDATION ERROR - FIX THE FRAMEWORK STRUCTURE
================================================================================

VALIDATION ERRORS FOUND:
{error_text}

{slot_summary}
{review_section}

YOUR TASK:
1. Keep the SAME slots with SAME purposes as listed above
2. Fix ONLY the structural/formatting errors (IDs, syntax, missing fields)
3. PRESERVE the original descriptions - DO NOT change them to generic text like "Complete X"
4. ALL slot IDs MUST be snake_case: lowercase letters, numbers, underscores ONLY
   BAD: "frontend html", "HTML Slot", "slot-a", "slot.b"
   GOOD: "frontend_html", "html_slot", "slot_a", "database_schema"
5. Framework needs: id, title, description, AND slots array
6. Each slot needs: id, title, description, persona, dependencies[], expected_outputs[]

EXAMPLE OF CORRECT FORMAT:
```json
{{
    "id": "framework_{timestamp}",
    "title": "Brief project title",
    "description": "What this builds",
    "slots": [
        {{
            "id": "frontend_html",
            "title": "Create Frontend",
            "description": "Build HTML/CSS/JS",
            "persona": "html_css_specialist",
            "dependencies": [],
            "expected_outputs": ["HTML file"]
        }}
    ]
}}
```

OUTPUT THE FIXED FRAMEWORK NOW:
```json
"""
        return base_prompt + fix_guidance
    
    async def decompose_from_scratch(
        self,
        query: str,
        personas: List[str],
        similarity_score: float = 0.0
    ) -> Optional[AdaptationResult]:
        """
        Decompose a query into a task framework from SCRATCH (no seed template).
        
        This is used when no template matches well (similarity < decomposition_similarity_threshold).
        The model generates a completely new framework based purely on the query.
        
        Args:
            query: The user query to decompose
            personas: List of available persona IDs for routing
            similarity_score: The similarity score of the best matching template
            
        Returns:
            AdaptationResult with the new framework, or None if failed
        """
        timestamp = int(time.time())
        original_id = f"scratch_{timestamp}"
        
        # Get the decompose_from_scratch prompt template
        if not self.decompose_from_scratch_prompt_template:
            logger.warning("No decompose_from_scratch_prompt template configured")
            return None
        
        # Build the prompt using replace() instead of format() to avoid curly brace conflicts
        prompt = self.decompose_from_scratch_prompt_template
        prompt = prompt.replace('{query}', query)
        prompt = prompt.replace('{timestamp}', str(timestamp))
        
        # Add persona context
        if self.config.get("persona_context_enabled", True):
            persona_context = self.get_persona_context_for_ids(personas)
            prompt = prompt.replace('{persona_context}', persona_context)
        else:
            simple_list = "\n".join(f"- {p}" for p in personas) if personas else "- python_backend"
            prompt = prompt.replace('{persona_context}', simple_list)
        
        # Add thinking mode prefix if configured
        if self.config.get("thinking_mode"):
            prompt = "<|im_start|>reasoning\n" + prompt
        
        logger.info(f"[TemplateAdapter] Decomposing from scratch (best template similarity: {similarity_score:.3f})")
        
        # DEBUG: Log the actual prompt being sent
        logger.info(f"[TemplateAdapter] Decompose prompt ({len(prompt)} chars):")
        logger.info("=" * 60)
        logger.info(prompt[:2000] + "..." if len(prompt) > 2000 else prompt)
        logger.info("=" * 60)
        
        debug_info = {
            "query": query,
            "mode": "decompose_from_scratch",
            "best_template_similarity": float(similarity_score),
            "attempts": []
        }
        
        try:
            for attempt in range(self.config.get("max_retries", 2)):
                attempt_info = {"attempt": attempt + 1, "timestamp": datetime.now().isoformat()}
                
                try:
                    # Generate with DeepSeek
                    logger.info(f"[TemplateAdapter] Attempt {attempt + 1}: Calling executor.generate()...")
                    response = await self._generate(prompt)
                    attempt_info["raw_response_length"] = len(response)
                    logger.info(f"[TemplateAdapter] Attempt {attempt + 1}: Got response ({len(response)} chars)")
                    logger.debug(f"[TemplateAdapter] Response preview: {response[:500]}...")
                    
                    # DEBUG: Log the raw response
                    logger.info(f"[TemplateAdapter] DeepSeek raw response ({len(response)} chars):")
                    logger.info("-" * 60)
                    logger.info(response[:3000] + "..." if len(response) > 3000 else response)
                    logger.info("-" * 60)
                    
                    # Parse response
                    parsed = parse_deepseek_response(response)
                    attempt_info["thinking_extracted"] = parsed.thinking is not None
                    attempt_info["rationale_extracted"] = parsed.rationale is not None
                    
                    if not parsed.success:
                        attempt_info["error"] = parsed.error
                        attempt_info["type"] = "parse_failure"
                        debug_info["attempts"].append(attempt_info)
                        
                        # Retry with modified prompt
                        prompt = self._generate_retry_prompt(prompt, attempt, parsed.error)
                        continue
                    
                    # Validate the generated framework
                    validation = self.validator.validate(parsed.json_data)
                    attempt_info["validation_passed"] = validation.is_valid
                    attempt_info["validation_errors"] = [str(e) for e in validation.errors]
                    
                    # Internal validation retry loop
                    internal_retries = self.config.get("internal_validation_retries", 1)
                    current_validation = validation
                    current_json_data = parsed.json_data
                    current_attempt = 0
                    
                    while not current_validation.is_valid and current_attempt < internal_retries:
                        logger.info(f"Validation failed in decompose_from_scratch - auto-retry {current_attempt + 1}/{internal_retries}")
                        
                        # Step 1: LLM Review - get fresh context review of the failure
                        review_result = await self._review_decomposition_with_llm(
                            failed_json=current_json_data,
                            validation_errors=current_validation.errors,
                            query=query
                        )
                        
                        # Step 2: Generate fix prompt with reviewer guidance
                        fix_prompt = self._generate_validation_fix_prompt(
                            prompt,
                            current_validation.errors,
                            original_id,
                            timestamp,
                            failed_json=current_json_data,
                            review_result=review_result
                        )
                        
                        # Step 3: Fresh generation with fix guidance
                        fix_response = await self._generate(fix_prompt)
                        fix_parsed = parse_deepseek_response(fix_response)
                        
                        if fix_parsed.success:
                            current_json_data = fix_parsed.json_data
                            current_validation = self.validator.validate(current_json_data)
                            attempt_info["internal_retry_success"] = True
                        
                        current_attempt += 1
                    
                    # Final validation check
                    if not current_validation.is_valid:
                        attempt_info["type"] = "validation_failure"
                        attempt_info["validation_errors"] = [str(e) for e in current_validation.errors]
                        debug_info["attempts"].append(attempt_info)
                        
                        # Retry
                        prompt = self._generate_retry_prompt(
                            prompt, 
                            attempt, 
                            "; ".join(str(e) for e in current_validation.errors)
                        )
                        continue
                    
                    # Success!
                    attempt_info["type"] = "success"
                    attempt_info["validation_passed"] = True
                    debug_info["attempts"].append(attempt_info)
                    
                    framework = TaskFramework.from_dict(current_json_data)
                    
                    logger.info(f"[TemplateAdapter] Decompose from scratch succeeded: {len(framework.slots)} slots")
                    
                    return AdaptationResult(
                        framework=framework,
                        rationale=parsed.rationale or parsed.thinking or "Decomposed from scratch - no template matched well",
                        adaptation_type="decompose_from_scratch",
                        attempts=attempt + 1,
                        validation_result=current_validation,
                        debug_info=debug_info
                    )
                    
                except Exception as e:
                    attempt_info["error"] = str(e)
                    attempt_info["type"] = "exception"
                    debug_info["attempts"].append(attempt_info)
                    logger.error(f"[TemplateAdapter] Attempt {attempt + 1} EXCEPTION: {type(e).__name__}: {e}")
                    import traceback
                    logger.error(f"[TemplateAdapter] Traceback: {traceback.format_exc()}")
                    
                    # Retry
                    prompt = self._generate_retry_prompt(prompt, attempt, str(e))
                    continue
            
            # All retries exhausted
            logger.error(f"[TemplateAdapter] Decompose from scratch failed after {self.config.get('max_retries', 2)} attempts")
            return None
            
        except Exception as e:
            logger.exception(f"[TemplateAdapter] Decompose from scratch exception: {e}")
            return None
    
    def _build_decompose_from_scratch_inline_prompt(
        self,
        query: str,
        personas_list: List[str],
        timestamp: int
    ) -> str:
        """Build an inline prompt when YAML template format fails."""
        # Get persona context
        if self.config.get("persona_context_enabled", True):
            persona_context = self.get_persona_context_for_ids(personas_list)
        else:
            persona_context = "\n".join(f"- {p}" for p in personas_list) if personas_list else "- python_backend"
        
        # Use replace() instead of format() to avoid curly brace conflicts
        prompt = """<|im_start|>user
You are an expert task decomposition system. Create a COMPLETE task framework from scratch based on this query.

## USER QUERY
{query}

## AVAILABLE PERSONAS
{persona_context}

## TASK
Decompose this query into 2-6 slots. For EACH slot, you MUST output a COMPLETE JSON framework with:
- Framework ID and title
- ALL slots with complete details (id, title, description, dependencies, expected_outputs)
- For each slot, specify the best persona_id from the available personas

## OUTPUT FORMAT
FRAMEWORK:
```json
{{
    "id": "framework_{timestamp}",
    "title": "[Title reflecting the query]",
    "description": "[Description]",
    "slots": [
        {{
            "id": "slot_name",
            "title": "Slot Title",
            "description": "Complete description of what this slot produces",
            "persona": "[best_matching_persona_id]",
            "dependencies": [],
            "expected_outputs": ["output type 1", "output type 2"]
        }}
    ]
}}
```

CRITICAL: Output MUST be complete JSON. Do not truncate.
<|im_end|>
<|im_start|>assistant
""".replace('{query}', query).replace('{persona_context}', persona_context).replace('{timestamp}', str(timestamp))
        return prompt
    
    async def _generate(self, prompt: str) -> str:
        """Generate response from DeepSeek."""
        # This method adapts to whatever executor interface is provided
        if hasattr(self.executor, 'generate_async'):
            return await self.executor.generate_async(
                prompt,
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"]
            )
        elif hasattr(self.executor, 'generate'):
            # Check if generate is async or sync
            import inspect
            if inspect.iscoroutinefunction(self.executor.generate):
                # Async executor - await directly
                return await self.executor.generate(
                    prompt,
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"]
                )
            else:
                # Sync executor, run in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self.executor.generate(
                        prompt,
                        temperature=self.config["temperature"],
                        max_tokens=self.config["max_tokens"]
                    )
                )
        else:
            raise RuntimeError("Executor must have generate() or generate_async() method")
    
    def create_single_slot_fallback(self, query: str) -> AdaptationResult:
        """
        Create a single-slot framework as catastrophic fallback.
        
        This is used when both adaptation AND seed template fail.
        """
        template = {
            "id": f"fallback_single_{int(time.time())}",
            "title": "Single-Slot Fallback Framework",
            "description": f"Fallback framework for: {query[:100]}...",
            "slots": [
                {
                    "id": "unified_response",
                    "title": "Unified Response",
                    "description": f"Answer the complete query: {query}",
                    "dependencies": [],
                    "expected_outputs": ["Complete response to user query"]
                }
            ]
        }
        
        framework = TaskFramework.from_dict(template)
        validation = self.validator.validate(template)
        
        return AdaptationResult(
            framework=framework,
            rationale="Catastrophic fallback - using single-slot framework",
            adaptation_type="single_slot_fallback",
            attempts=0,
            validation_result=validation,
            debug_info={"fallback_reason": "catastrophic_failure"}
        )


class MockExecutor:
    """Mock executor for testing."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Return a mock response."""
        return """<|im_start|>reasoning
Analyzing the query about authentication...
The user wants JWT auth with user registration.
I'll adapt the template by adding a registration slot.



RATIONALE:
Added user_registration slot since the query explicitly mentions registration.
Kept the core JWT authentication slots.

ADAPTED FRAMEWORK:
```json
{
    "id": "adapted_auth_template_12345",
    "title": "JWT Authentication with Registration",
    "description": "Complete auth system with user registration",
    "slots": [
        {
            "id": "user_model",
            "title": "User Data Model",
            "description": "Define user schema with password hashing",
            "dependencies": [],
            "expected_outputs": ["User model class", "Password utilities"]
        },
        {
            "id": "user_registration",
            "title": "User Registration",
            "description": "Implement user signup endpoint",
            "dependencies": ["user_model"],
            "expected_outputs": ["Registration endpoint", "Validation logic"]
        },
        {
            "id": "jwt_utils",
            "title": "JWT Utilities",
            "description": "Token creation and validation",
            "dependencies": [],
            "expected_outputs": ["JWT encode/decode functions"]
        },
        {
            "id": "login_endpoint",
            "title": "Login Endpoint",
            "description": "Authenticate user and issue tokens",
            "dependencies": ["user_model", "jwt_utils"],
            "expected_outputs": ["Login route", "Token response"]
        }
    ]
}
```"""


async def test_adapter():
    """Test the adapter with persona-aware prompt generation."""
    logging.basicConfig(level=logging.DEBUG)
    
    # Create mock executor
    executor = MockExecutor()
    validator = TemplateValidator()
    
    # Load the persona-aware prompt template
    adapter = TemplateAdapter(
        executor, 
        validator,
        prompt_template_path="config/prompts/template_adapter.yaml"
    )
    
    print("=" * 60)
    print("Template Adapter Test (Persona-Aware)")
    print("=" * 60)
    
    # Test 1: Check persona loading
    print(f"\n1. Persona Context Loaded:")
    print(f"   Total personas: {len(adapter.personas)}")
    for pid in list(adapter.personas.keys())[:5]:
        print(f"   - {pid}: {adapter.personas[pid].get('display_name', pid)}")
    
    # Test 2: Check persona context text
    print(f"\n2. Persona Context Text:")
    context = adapter._persona_context_text
    print(f"   Length: {len(context)} characters")
    print(f"   Preview: {context[:200]}...")
    
    # Test 3: Generate slot prompt for a specific persona
    print(f"\n3. Slot Prompt Generation (python_backend):")
    slot_prompt = adapter.generate_slot_prompt(
        slot_title="Create User API",
        slot_description="Build FastAPI endpoint for user CRUD operations",
        expected_outputs=["FastAPI router code", "Pydantic models"],
        persona_id="python_backend"
    )
    print(f"   Generated prompt preview:\n{slot_prompt[:300]}...")
    
    # Test 4: Adapt template with persona context
    print(f"\n4. Template Adaptation:")
    seed = {
        "id": "auth_template",
        "title": "Authentication Template",
        "description": "Basic auth",
        "slots": [
            {"id": "user_model", "title": "User Model", "dependencies": []},
            {"id": "jwt_utils", "title": "JWT", "dependencies": []},
            {"id": "login", "title": "Login", "dependencies": ["user_model", "jwt_utils"]}
        ]
    }
    
    query = "Build a FastAPI authentication system with JWT tokens and user registration"
    personas = ["python_backend", "security_architect", "sql_query_builder"]
    
    result = await adapter.adapt(query, seed, personas)
    
    print(f"   Adaptation Type: {result.adaptation_type}")
    print(f"   Attempts: {result.attempts}")
    print(f"   Valid: {result.validation_result.is_valid}")
    print(f"   Framework ID: {result.framework.id}")
    print(f"   Slots: {len(result.framework.slots)}")
    
    # Show slot assignments
    print(f"\n   Slot Persona Assignments:")
    for slot in result.framework.slots:
        print(f"   - {slot.id}: {slot.persona or 'unassigned'}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_adapter())
