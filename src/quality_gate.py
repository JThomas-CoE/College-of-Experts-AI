"""
V12 Quality Gate - LLM-Based Reviewer System
College of Experts Architecture

Key Design:
1. Uses SAME savant model that generated the output (no reload)
2. Applies NEW reviewer persona (fresh context, not generator's persona)
3. LLM-based evaluation (not rule-based)
4. Provides: pass/fail decision, issues, and refined prompt for retry
"""

import re
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any


# =============================================================================
# REVIEWER PERSONA LOADER
# =============================================================================

class ReviewerPersonaLoader:
    """Loads reviewer persona harnesses from YAML files."""
    
    def __init__(self, prompts_dir: str = "config/prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._personas = {}
        self._load_personas()
    
    def _load_personas(self):
        """Load all reviewer personas from YAML files."""
        persona_files = {
            "python": "python_reviewer.yaml",
            "html": "html_reviewer.yaml",
            "sql": "sql_reviewer.yaml",
            "general": "general_reviewer.yaml"
        }
        
        for code_type, filename in persona_files.items():
            filepath = self.prompts_dir / filename
            if filepath.exists():
                with open(filepath) as f:
                    persona_data = yaml.safe_load(f)
                    self._personas[code_type] = persona_data
            else:
                # Fallback to hardcoded persona if file not found
                self._personas[code_type] = self._get_fallback_persona(code_type)
    
    def _get_fallback_persona(self, code_type: str) -> Dict:
        """Fallback hardcoded personas if YAML files not found."""
        personas = {
            "python": {
                "persona": """You are an EXPERT QUALITY and COMPLETENESS REVIEWER for Python code.

Your role is to verify that Python output is:
- Complete and runnable (all imports present, no undefined references)
- Properly structured (no truncated functions/classes)
- Production-ready (no "..." or "TODO" placeholders indicating missing code)

Check for:
1. All imports at module start (from X import Y, import Z)
2. App initialization (app = Flask(__name__) or app = FastAPI())
3. Route handlers with actual logic, not empty functions
4. No incomplete function definitions (def foo(:
5. All code blocks properly closed with ```
6. No mid-statement truncation
7. Model classes fully defined with all fields
8. Database models have all required fields and relationships

FAIL the output if any critical elements are missing or if code appears truncated.""",
                "output_format": """DECISION: [PASS or FAIL]
If PASS:
FEEDBACK: [Brief positive feedback]
If FAIL:
ISSUES:
- [Issue 1]
- [Issue 2]
- ...

SUGGESTIONS:
- [Suggestion 1]
- [Suggestion 2]
- ...

REFINED_PROMPT:
[Improved prompt addressing the issues above]"""
            },
            "html": {
                "persona": """You are an EXPERT QUALITY and COMPLETENESS REVIEWER for HTML/CSS/JavaScript code.

Your role is to verify that HTML output is:
- Complete and self-contained (all CSS/JS embedded or properly linked)
- Semantically correct (proper use of HTML5 elements)
- Well-structured (DOCTYPE, html, head, body tags present)
- Production-ready (no truncation, no placeholder comments indicating missing code)

Check for:
1. Complete document structure with <!DOCTYPE html>, <html>, <head>, <body>
2. All CSS in <style> tags or proper <link> references
3. All JavaScript in <script> tags or proper src references
4. No "TODO" or "MISSING" or "..." placeholders
5. All tags properly opened and closed
6. Forms have action attributes and proper input types
7. No truncated code blocks (``` without closing ```)

FAIL the output if any required elements are missing or if code appears incomplete.""",
                "output_format": """DECISION: [PASS or FAIL]
If PASS:
FEEDBACK: [Brief positive feedback]
If FAIL:
ISSUES:
- [Issue 1]
- [Issue 2]
- ...

SUGGESTIONS:
- [Suggestion 1]
- [Suggestion 2]
- ...

REFINED_PROMPT:
[Improved prompt addressing the issues above]"""
            },
            "sql": {
                "persona": """You are an EXPERT QUALITY and COMPLETENESS REVIEWER for SQL/DDL code.

Your role is to verify that SQL output is:
- Complete and executable (no truncated CREATE TABLE statements)
- Properly structured (all parentheses balanced)
- Production-ready (no "..." placeholders)

Check for:
1. CREATE TABLE statements have all columns defined
2. All parentheses properly balanced
3. PRIMARY KEY and FOREIGN KEY constraints present
4. No truncated statements ending with "..."
5. All code blocks properly closed with ```
6. Indexes defined for performance
7. Data types specified for all columns

FAIL the output if any required schema elements are missing or if DDL appears incomplete.""",
                "output_format": """DECISION: [PASS or FAIL]
If PASS:
FEEDBACK: [Brief positive feedback]
If FAIL:
ISSUES:
- [Issue 1]
- [Issue 2]
- ...

SUGGESTIONS:
- [Suggestion 1]
- [Suggestion 2]
- ...

REFINED_PROMPT:
[Improved prompt addressing the issues above]"""
            },
            "general": {
                "persona": """You are an EXPERT CONTENT REVIEWER.

Your role is to verify that text output is:
- Complete and addresses all parts of the task
- Accurate and well-supported
- Clear and well-organized
- Free of hallucinations and unsupported claims

Check for:
1. All requested sections present
2. No fabricated statistics or claims
3. Clear structure with headings
4. No truncation indicators

FAIL only if critical requirements are not met.""",
                "output_format": """DECISION: [PASS or FAIL]
If PASS:
FEEDBACK: [Brief positive feedback]
If FAIL:
ISSUES:
- [Issue 1]
- [Issue 2]
- ...

SUGGESTIONS:
- [Suggestion 1]
- [Suggestion 2]
- ...

REFINED_PROMPT:
[Improved prompt addressing the issues above]"""
            }
        }
        return personas[code_type]
    
    def get_persona(self, code_type: str) -> str:
        """Get reviewer persona for given code type."""
        return self._personas.get(code_type, self._get_fallback_persona(code_type))["persona"]
    
    def get_output_format(self, code_type: str) -> str:
        """Get expected output format for reviewer."""
        return self._personas.get(code_type, self._get_fallback_persona(code_type))["output_format"]


# =============================================================================
# VALIDATION RESULT
# =============================================================================

@dataclass
class ValidationResult:
    """Result of quality validation."""
    passed: bool
    summary: str
    issues: List[str]
    completeness_score: int  # 0-10
    accuracy_score: int      # 0-10
    truncated: bool
    reviewer_persona: str    # The reviewer persona used
    code_type: str           # The type of code reviewed (python/sql/html/unknown)
    refined_prompt: str = ""  # Refined prompt from reviewer for retry


# =============================================================================
# QUALITY GATE - LLM-BASED
# =============================================================================

class QualityGate:
    """
    LLM-based quality validation with reviewer persona.
    
    Key Design:
    1. Uses SAME savant model that generated output (no reload)
    2. Applies NEW reviewer persona (fresh context)
    3. LLM evaluates output and provides: pass/fail, issues, refined prompt
    4. Max 3 retries before fallback to DeepSeek
    """
    
    PASS_THRESHOLD = 9
    RETRY_THRESHOLD = 4
    
    def __init__(self, prompts_dir: str = "config/prompts"):
        self.persona_loader = ReviewerPersonaLoader(prompts_dir)
    
    def _detect_code_type(self, expected_outputs: List[str]) -> str:
        """Detect what type of code is expected."""
        text = " ".join(expected_outputs).lower() if expected_outputs else ""
        
        if any(kw in text for kw in ["python", "flask", "fastapi", "module", "backend"]):
            return "python"
        elif any(kw in text for kw in ["sql", "ddl", "schema", "database"]):
            return "sql"
        elif any(kw in text for kw in ["html", "frontend", "webpage", "interface"]):
            return "html"
        return "general"
    
    def _build_reviewer_prompt(self, code_type: str, slot_description: str, expert_output: str, expert_persona: Optional[str] = None) -> str:
        """Build prompt for reviewer LLM with persona awareness."""
        persona = self.persona_loader.get_persona(code_type)
        
        # Add persona context if available
        persona_context = ""
        if expert_persona:
            persona_context = f"""

## GENERATOR EXPERT CONTEXT
The code above was generated by the following expert persona:

{expert_persona}

As a reviewer, consider:
- Does the output reflect the expertise expected from this expert?
- Are there domain-specific considerations this expert should have addressed?
- Is the quality consistent with what this expert should produce?
"""
        
        return f"""{persona}

## TASK DESCRIPTION
{slot_description}
{persona_context}
## OUTPUT TO REVIEW
{expert_output}

## REVIEW INSTRUCTIONS
Evaluate the output and provide your assessment in the following format:

{self.persona_loader.get_output_format(code_type)}"""
    
    def _parse_reviewer_response(self, review_text: str) -> Tuple[bool, List[str], str]:
        """
        Parse reviewer's response to extract decision, issues, and refined prompt.
        
        Returns:
            (passed, issues, refined_prompt)
        """
        # Extract decision
        decision_match = re.search(r'DECISION:\s*(PASS|FAIL)', review_text, re.IGNORECASE)
        passed = decision_match and decision_match.group(1).upper() == "PASS"
        
        # Extract issues
        issues = []
        issues_section = re.search(r'ISSUES:(.*?)(?:SUGGESTIONS:|REFINED_PROMPT:|$)', review_text, re.IGNORECASE | re.DOTALL)
        if issues_section:
            for line in issues_section.group(1).strip().split('\n'):
                line = line.strip().lstrip('-*')
                if line:
                    issues.append(line)
        
        # Extract refined prompt
        refined_prompt = ""
        refined_section = re.search(r'REFINED_PROMPT:(.*?)(?:DECISION:|$)', review_text, re.IGNORECASE | re.DOTALL)
        if refined_section:
            refined_prompt = refined_section.group(1).strip()
        
        return passed, issues, refined_prompt
    
    async def validate_with_llm(
        self, 
        executor: Any,
        content: str, 
        slot: Any, 
        expert_id: str,
        slot_id: str = None
    ) -> ValidationResult:
        """
        Validate slot output using LLM-based reviewer.
        
        Args:
            executor: The model executor (with loaded savant model)
            content: The output to validate
            slot: The slot being validated
            expert_id: The expert that generated the output
            slot_id: The executor slot ID to use (ensures correct model is used)
        
        Returns:
            ValidationResult with pass/fail decision, issues, and scores
        """
        # =====================================================================
        # BASIC SANITY CHECKS (keep these - they're fast and reliable)
        # =====================================================================
        
        # Refusal detection
        refusal_phrases = [
            "i'm sorry, but i can't",
            "i cannot assist with that",
            "i'm unable to help with",
            "i can't help with that",
            "as an ai",
            "i apologize, but",
            "i'm not able to",
            "unfortunately, i cannot",
            "i don't have the ability",
            "beyond my capabilities",
            "i must decline",
            "i won't be able to",
            "i'm afraid i can't",
        ]
        content_lower = content.lower()
        for phrase in refusal_phrases:
            if phrase in content_lower:
                return ValidationResult(
                    passed=False,
                    summary="Model refused to answer",
                    issues=[
                        f"REFUSAL: '{phrase}' detected",
                        "-> IMPROVE: Rephrase the task to be more specific and within domain scope."
                    ],
                    completeness_score=0,
                    accuracy_score=0,
                    truncated=False,
                    reviewer_persona="",
                    code_type=""
                )
        
        # Minimum substance check
        if not content or len(content.strip()) < 50:
            return ValidationResult(
                passed=False,
                summary="Output too short",
                issues=[
                    f"LENGTH: {len(content)} chars (minimum 50 required)",
                    f"-> IMPROVE: Provide a meaningful response for '{slot.title}', not a stub."
                ],
                completeness_score=1,
                accuracy_score=1,
                truncated=False,
                reviewer_persona="",
                code_type=""
            )
        
        # CRITICAL: Repetition detection (LLM echoing prompts or context)
        # Check for repeated blocks of text (indicates degenerate output)
        lines = content.split('\n')
        if len(lines) > 10:
            # Check for repeated patterns
            line_counts = {}
            for line in lines:
                line_stripped = line.strip()
                if len(line_stripped) > 20:  # Only count substantial lines
                    line_counts[line_stripped] = line_counts.get(line_stripped, 0) + 1
            
            # If any line appears 5+ times, it's degenerate repetition
            for line, count in line_counts.items():
                if count >= 5:
                    return ValidationResult(
                        passed=False,
                        summary="Degenerate repetition detected",
                        issues=[
                            f"REPETITION: Line repeated {count} times: '{line[:50]}...'",
                            "-> IMPROVE: Generate unique content, not repeated patterns."
                        ],
                        completeness_score=0,
                        accuracy_score=0,
                        truncated=False,
                        reviewer_persona="",
                        code_type=""
                    )
        
        # CRITICAL: Placeholder detection (empty template content)
        placeholder_patterns = [
            "/* Your CSS code here */",
            "/* Your code here */",
            "/* TODO */",
            "<!-- Your HTML code here -->",
            "<!-- TODO -->",
            "# Your code here",
            "# TODO: implement",
            "pass  # TODO",
        ]
        content_lower = content.lower()
        for placeholder in placeholder_patterns:
            if placeholder.lower() in content_lower:
                return ValidationResult(
                    passed=False,
                    summary="Placeholder content detected",
                    issues=[
                        f"PLACEHOLDER: '{placeholder}' found - code not implemented",
                        "-> IMPROVE: Replace placeholders with actual implementation."
                    ],
                    completeness_score=2,
                    accuracy_score=0,
                    truncated=False,
                    reviewer_persona="",
                    code_type=""
                )
        
        # =====================================================================
        # LLM-BASED REVIEW
        # =====================================================================
        
        # Detect code type for reviewer persona selection
        code_type = self._detect_code_type(slot.expected_outputs or [])
        
        # Build reviewer prompt
        reviewer_prompt = self._build_reviewer_prompt(
            code_type, 
            slot.description, 
            content
        )
        
        # Call executor with reviewer prompt (same model, new persona)
        # Note: Pass slot_id to ensure we use the already-loaded model (no VRAM spike)
        try:
            review_response = await executor.generate(
                reviewer_prompt,
                max_tokens=4096,  # Reviewer needs enough output space
                temperature=0.3,  # Lower temp for consistent evaluation
                slot_id=slot_id   # VRAM-tracked: uses existing loaded model
            )
        except Exception as e:
            # If LLM call fails, return conservative result
            return ValidationResult(
                passed=False,
                summary=f"Reviewer LLM error: {e}",
                issues=[f"LLM ERROR: {e}"],
                completeness_score=5,
                accuracy_score=5,
                truncated=False,
                reviewer_persona=self.persona_loader.get_persona(code_type),
                code_type=code_type
            )
        
        # Parse reviewer response
        passed, issues, refined_prompt = self._parse_reviewer_response(review_response)
        
        # Calculate scores based on reviewer decision
        if passed:
            completeness_score = 10
            accuracy_score = 10
            summary = f"Valid ({len(content)} chars)"
        else:
            # Score based on number of issues
            issue_count = len(issues)
            completeness_score = max(0, 10 - issue_count)
            accuracy_score = max(0, 10 - issue_count)
            summary = f"REJECTED: {issues[0] if issues else 'Failed review'}"
        
        # OVERRIDE: If scores are perfect OR meet threshold, pass regardless of DECISION parsing
        # This resolves the "9/9 FAIL" contradiction where 1 minor issue shouldn't force a retry
        if (completeness_score >= self.PASS_THRESHOLD and accuracy_score >= self.PASS_THRESHOLD) or \
           (completeness_score == 10 and accuracy_score == 10):
            passed = True
            summary = f"Valid (score {completeness_score}/{self.PASS_THRESHOLD})"
            # We keep the issues list so the user can see them, even if passed
        
        return ValidationResult(
            passed=passed,
            summary=summary,
            issues=issues,
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            truncated=False,  # LLM-based review doesn't detect truncation directly
            reviewer_persona=self.persona_loader.get_persona(code_type),
            code_type=code_type,
            refined_prompt=refined_prompt  # Include refined prompt from reviewer
        )
    
    def generate_refined_prompt(
        self, 
        base_prompt: str, 
        slot: Any, 
        issues: List[str],
        refined_prompt_from_reviewer: str
    ) -> str:
        """
        Generate refined prompt for retry.
        
        Args:
            base_prompt: The original prompt
            slot: The slot being executed
            issues: Issues from quality gate
            refined_prompt_from_reviewer: Refined prompt from reviewer LLM
        
        Returns:
            Refined prompt with helpful suggestions
        """
        # If reviewer provided a refined prompt, use it
        if refined_prompt_from_reviewer:
            # Inject reviewer's refined prompt into base prompt
            user_marker = "<|im_start|>user\n"
            if user_marker in base_prompt:
                parts = base_prompt.split(user_marker, 1)
                refined = parts[0] + user_marker + refined_prompt_from_reviewer + "\n\n" + parts[1]
            else:
                refined = refined_prompt_from_reviewer + "\n\n" + base_prompt
            return refined
        
        # Fallback: generate refined prompt from issues
        requirements = []
        for issue in issues:
            issue_lower = issue.lower()
            
            # Truncation
            if "truncation" in issue_lower:
                requirements.append("Ensure all code blocks are complete and properly closed with ```")
                requirements.append("Complete all function/class definitions before ending output.")
            
            # Missing elements
            elif "missing" in issue_lower or "incomplete" in issue_lower:
                if "flask" in issue_lower or "fastapi" in issue_lower:
                    requirements.append("Include Flask/FastAPI imports and app initialization.")
                if "route" in issue_lower or "endpoint" in issue_lower:
                    requirements.append("Implement complete API routes with actual logic.")
                if "auth" in issue_lower or "password" in issue_lower:
                    requirements.append("Include authentication logic with bcrypt password hashing.")
            
            # Syntax errors
            elif "syntax" in issue_lower:
                requirements.append("Ensure code is syntactically correct with proper indentation (4 spaces).")
                requirements.append("All brackets ((), [], {}) must be balanced.")
            
            # General completeness
            else:
                requirements.append("Provide complete, detailed implementation with all required elements.")
        
        if not requirements:
            requirements = [
                "Provide complete, detailed implementation with all required elements.",
                "Include actual code in appropriate code blocks.",
                "Ensure all requested features are implemented."
            ]
        
        # Deduplicate
        seen = set()
        unique_reqs = []
        for req in requirements:
            req_lower = req.lower()
            if req_lower not in seen:
                seen.add(req_lower)
                unique_reqs.append(req)
        
        unique_reqs = unique_reqs[:8]
        
        # Inject requirements
        req_list = '\n'.join(f'- {req}' for req in unique_reqs)
        requirements_block = f"""## REVIEWER REMINDERS - Address these issues:
{req_list}

IMPORTANT: Ensure all reminders above are addressed before responding.

---

"""
        
        user_marker = "<|im_start|>user\n"
        if user_marker in base_prompt:
            parts = base_prompt.split(user_marker, 1)
            refined = parts[0] + user_marker + requirements_block + parts[1]
        else:
            refined = requirements_block + base_prompt
        
        return refined
