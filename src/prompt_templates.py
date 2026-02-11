"""
Prompt Templates Module for College of Experts

This module provides template functions for building persona-aware prompts
that incorporate expert identity, role, expertise areas, communication style,
and constraints into the prompt construction process.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml

from .persona_context import PersonaContext, get_persona_for_expert


class PromptTemplates:
    """Template builder for persona-aware prompts."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize prompt templates.
        
        Args:
            config_dir: Directory containing prompt configuration files
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config" / "prompts"
        self.config_dir = Path(config_dir)
        self._templates: Dict[str, Dict] = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load prompt templates from configuration files."""
        if not self.config_dir.exists():
            return
        
        for template_file in self.config_dir.glob("*.yaml"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = yaml.safe_load(f)
                    if template_data:
                        self._templates[template_file.stem] = template_data
            except Exception as e:
                print(f"Warning: Failed to load template {template_file}: {e}")
    
    def build_system_prompt(
        self,
        persona: PersonaContext,
        task_context: Optional[Dict[str, Any]] = None,
        include_knowledge: bool = True
    ) -> str:
        """
        Build a comprehensive system prompt incorporating persona information.
        
        Args:
            persona: The persona context for the expert
            task_context: Additional context about the current task
            include_knowledge: Whether to include knowledge layer information
            
        Returns:
            A formatted system prompt string
        """
        sections = []
        
        # Identity and Role
        sections.append(self._build_identity_section(persona))
        
        # Expertise Areas
        sections.append(self._build_expertise_section(persona))
        
        # Communication Style
        sections.append(self._build_communication_section(persona))
        
        # Constraints and Guidelines
        sections.append(self._build_constraints_section(persona))
        
        # Task-specific context
        if task_context:
            sections.append(self._build_task_context_section(task_context))
        
        # Knowledge Layer
        if include_knowledge and persona.knowledge_layer:
            sections.append(self._build_knowledge_section(persona))
        
        # Output Format
        sections.append(self._build_output_format_section(persona))
        
        return "\n\n".join(sections)
    
    def _build_identity_section(self, persona: PersonaContext) -> str:
        """Build the identity section of the prompt."""
        lines = [
            f"You are {persona.name}, a {persona.role}.",
            f"Expert ID: {persona.expert_id}",
            f"Expert Type: {persona.expert_type}"
        ]
        
        if persona.description:
            lines.append(f"\n{persona.description}")
        
        return "\n".join(lines)
    
    def _build_expertise_section(self, persona: PersonaContext) -> str:
        """Build the expertise areas section of the prompt."""
        lines = ["Your areas of expertise:"]
        
        if persona.expertise_areas:
            for area in persona.expertise_areas:
                lines.append(f"  - {area}")
        
        if persona.specializations:
            lines.append("\nSpecializations:")
            for spec in persona.specializations:
                lines.append(f"  - {spec}")
        
        if persona.capabilities:
            lines.append("\nCapabilities:")
            for cap in persona.capabilities:
                lines.append(f"  - {cap}")
        
        return "\n".join(lines)
    
    def _build_communication_section(self, persona: PersonaContext) -> str:
        """Build the communication style section of the prompt."""
        lines = ["Communication style:"]
        
        if persona.communication_style:
            lines.append(f"Style: {persona.communication_style}")
        
        if persona.tone:
            lines.append(f"Tone: {persona.tone}")
        
        if persona.response_format:
            lines.append(f"Response Format: {persona.response_format}")
        
        if persona.examples:
            lines.append("\nExample Responses:")
            for i, example in enumerate(persona.examples, 1):
                lines.append(f"\nExample {i}:")
                lines.append(f"  {example}")
        
        return "\n".join(lines)
    
    def _build_constraints_section(self, persona: PersonaContext) -> str:
        """Build the constraints and guidelines section of the prompt."""
        lines = []
        
        if persona.constraints:
            lines.append("Constraints:")
            for constraint in persona.constraints:
                lines.append(f"  - {constraint}")
        
        if persona.guidelines:
            lines.append("\nGuidelines:")
            for guideline in persona.guidelines:
                lines.append(f"  - {guideline}")
        
        if persona.limitations:
            lines.append("\nLimitations:")
            for limitation in persona.limitations:
                lines.append(f"  - {limitation}")
        
        return "\n".join(lines) if lines else ""
    
    def _build_task_context_section(self, task_context: Dict[str, Any]) -> str:
        """Build the task-specific context section."""
        lines = ["Task context:"]
        
        for key, value in task_context.items():
            if isinstance(value, list):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
            elif isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
    
    def _build_knowledge_section(self, persona: PersonaContext) -> str:
        """Build the knowledge layer section."""
        lines = []
        
        if persona.knowledge_layer:
            if persona.knowledge_layer.get('domains'):
                lines.append("Knowledge domains:")
                for domain in persona.knowledge_layer['domains']:
                    lines.append(f"  - {domain}")
            
            if persona.knowledge_layer.get('concepts'):
                lines.append("\nKey concepts:")
                for concept in persona.knowledge_layer['concepts']:
                    lines.append(f"  - {concept}")
            
            if persona.knowledge_layer.get('resources'):
                lines.append("\nAvailable resources:")
                for resource in persona.knowledge_layer['resources']:
                    lines.append(f"  - {resource}")
        
        return "\n".join(lines) if lines else ""
    
    def _build_output_format_section(self, persona: PersonaContext) -> str:
        """Build the output format section."""
        lines = []
        
        if persona.output_format:
            lines.append(f"Output format: {persona.output_format}")
        
        if persona.response_format:
            lines.append(f"Response style: {persona.response_format}")
        
        lines.append("\nProvide clear, well-structured responses that reflect your expertise.")
        
        return "\n".join(lines)
    
    def build_user_prompt(
        self,
        query: str,
        persona: Optional[PersonaContext] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build a user prompt with optional persona context.
        
        Args:
            query: The user's query or task
            persona: Optional persona context for customization
            context: Additional context information
            
        Returns:
            A formatted user prompt string
        """
        prompt_parts = []
        
        if context:
            prompt_parts.append(self._format_context(context))
        
        prompt_parts.append(query)
        
        if persona and persona.response_format:
            prompt_parts.append(f"\n\nPlease respond in the following format: {persona.response_format}")
        
        return "\n\n".join(prompt_parts)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context information for the prompt."""
        lines = ["Context:"]
        
        for key, value in context.items():
            if isinstance(value, (list, dict)):
                lines.append(f"{key}:")
                lines.append(f"  {value}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
    
    def build_slot_prompt(
        self,
        slot_name: str,
        query: str,
        persona: PersonaContext,
        task_context: Optional[Dict[str, Any]] = None,
        template_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Build a complete prompt for a specific expert slot.
        
        Args:
            slot_name: Name of the expert slot
            query: The user's query or task
            persona: The persona context for this expert
            task_context: Additional task context
            template_name: Optional specific template to use
            
        Returns:
            Dictionary with 'system' and 'user' prompt keys
        """
        # Load template-specific configuration if provided
        template_config = None
        if template_name and template_name in self._templates:
            template_config = self._templates[template_name]
        
        # Build system prompt
        system_prompt = self.build_system_prompt(
            persona=persona,
            task_context=task_context,
            include_knowledge=True
        )
        
        # Apply template-specific modifications if available
        if template_config:
            system_prompt = self._apply_template_config(
                system_prompt, 
                template_config,
                slot_name
            )
        
        # Build user prompt
        user_prompt = self.build_user_prompt(
            query=query,
            persona=persona,
            context=task_context
        )
        
        return {
            'system': system_prompt,
            'user': user_prompt,
            'slot': slot_name,
            'expert_id': persona.expert_id
        }
    
    def _apply_template_config(
        self,
        base_prompt: str,
        template_config: Dict,
        slot_name: str
    ) -> str:
        """
        Apply template-specific configuration to the base prompt.
        
        Args:
            base_prompt: The base system prompt
            template_config: Template configuration
            slot_name: Name of the slot
            
        Returns:
            Modified prompt
        """
        prompt = base_prompt
        
        # Add template-specific prefix
        if 'prefix' in template_config:
            prompt = f"{template_config['prefix']}\n\n{prompt}"
        
        # Add template-specific suffix
        if 'suffix' in template_config:
            prompt = f"{prompt}\n\n{template_config['suffix']}"
        
        # Add slot-specific instructions
        if 'slot_instructions' in template_config:
            slot_instructions = template_config['slot_instructions'].get(slot_name)
            if slot_instructions:
                prompt = f"{prompt}\n\nAdditional instructions for this task:\n{slot_instructions}"
        
        return prompt
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template names."""
        return list(self._templates.keys())
    
    def get_template(self, template_name: str) -> Optional[Dict]:
        """Get a specific template configuration."""
        return self._templates.get(template_name)


# Convenience functions for common use cases

def build_persona_aware_prompt(
    persona: PersonaContext,
    task_description: str,
    project_context: str = "",
    output_formats: Optional[str] = None,
    dependency_signatures: Optional[str] = None,
    knowledge_context: str = "",
    compact_mode: bool = False,
    context_budget: int = 4096
) -> str:
    """
    Build a persona-aware prompt for an expert slot.
    
    Args:
        persona: The persona context for the expert
        task_description: Description of the task for this slot
        project_context: Overall project context/description
        output_formats: Expected output formats for this slot
        dependency_signatures: Signatures from dependent slots (if any)
        knowledge_context: Knowledge base context to include
        compact_mode: Whether to use compact formatting for smaller context budgets
        context_budget: Available context window in tokens
        
    Returns:
        A formatted prompt string
    """
    sections = []
    
    # Identity section
    sections.append(f"You are {persona.name}, a {persona.role}.")
    
    if persona.background:
        sections.append(f"\nBackground: {persona.background}")
    
    # Project context
    if project_context:
        sections.append(f"\nPROJECT CONTEXT:")
        sections.append(f"Overall project: {project_context}")
    
    # Expertise
    if persona.expertise:
        sections.append(f"\nYour areas of expertise:")
        for area in persona.expertise:
            sections.append(f"- {area}")
    
    # Task description
    sections.append(f"\nYOUR TASK:")
    sections.append(task_description)
    
    # Output formats
    if output_formats:
        sections.append(f"\nExpected output format: {output_formats}")
    
    # Knowledge context
    if knowledge_context:
        sections.append(f"\nReference knowledge: {knowledge_context}")
    
    # Dependency signatures
    if dependency_signatures:
        sections.append(f"\nBuild upon these completed sections:")
        sections.append(dependency_signatures)
    
    # Constraints and guidelines
    if persona.constraints:
        sections.append(f"\nImportant constraints:")
        for constraint in persona.constraints:
            sections.append(f"- {constraint}")
    
    if persona.guidelines:
        sections.append(f"\nGuidelines to follow:")
        for guideline in persona.guidelines:
            sections.append(f"- {guideline}")
    
    # Communication style
    if persona.communication_style:
        sections.append(f"\nCommunication style: {persona.communication_style}")
    
    if persona.tone:
        sections.append(f"Tone: {persona.tone}")
    
    # Compact mode adjustments
    if compact_mode:
        # Remove extra whitespace for compact mode
        prompt = "\n".join(sections)
    else:
        prompt = "\n\n".join(sections)
    
    return prompt


def build_persona_prompt(
    expert_id: str,
    expert_type: str,
    query: str,
    config_dir: Optional[Path] = None,
    task_context: Optional[Dict[str, Any]] = None,
    template_name: Optional[str] = None
) -> Dict[str, str]:
    """
    Convenience function to build a persona-aware prompt.
    
    Args:
        expert_id: Unique identifier for the expert
        expert_type: Type/category of the expert
        query: The user's query or task
        config_dir: Directory containing configuration files
        task_context: Additional task context
        template_name: Optional specific template to use
        
    Returns:
        Dictionary with 'system' and 'user' prompt keys
    """
    # Load persona context
    persona = get_persona_for_expert(
        expert_id=expert_id,
        expert_type=expert_type,
        config_dir=config_dir
    )
    
    # Build prompts
    templates = PromptTemplates(config_dir=config_dir)
    return templates.build_slot_prompt(
        slot_name=expert_id,
        query=query,
        persona=persona,
        task_context=task_context,
        template_name=template_name
    )


def build_multi_expert_prompts(
    experts: List[Dict[str, str]],
    query: str,
    config_dir: Optional[Path] = None,
    task_context: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    """
    Build prompts for multiple experts.
    
    Args:
        experts: List of expert dictionaries with 'expert_id' and 'expert_type'
        query: The user's query or task
        config_dir: Directory containing configuration files
        task_context: Additional task context
        
    Returns:
        List of prompt dictionaries
    """
    prompts = []
    templates = PromptTemplates(config_dir=config_dir)
    
    for expert in experts:
        persona = get_persona_for_expert(
            expert_id=expert['expert_id'],
            expert_type=expert['expert_type'],
            config_dir=config_dir
        )
        
        prompt = templates.build_slot_prompt(
            slot_name=expert['expert_id'],
            query=query,
            persona=persona,
            task_context=task_context
        )
        
        prompts.append(prompt)
    
    return prompts
