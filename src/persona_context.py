"""
Persona Context Loader Module

This module provides utilities for loading and managing persona-aware context
for the College of Experts system. It handles loading persona configurations
from YAML files and building comprehensive persona context for prompts.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class PersonaContext:
    """Complete persona context for an expert."""
    name: str
    role: str
    expertise: List[str]
    tone: str
    perspective: str
    constraints: List[str]
    examples: List[str]
    background: str = ""
    communication_style: str = ""
    key_principles: List[str] = field(default_factory=list)
    # Additional attributes for PromptTemplates compatibility
    expert_id: str = ""
    expert_type: str = ""
    description: str = ""
    expertise_areas: List[str] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    response_format: str = ""
    guidelines: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    output_format: str = ""
    knowledge_layer: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert persona context to dictionary."""
        return {
            "name": self.name,
            "role": self.role,
            "expertise": self.expertise,
            "tone": self.tone,
            "perspective": self.perspective,
            "constraints": self.constraints,
            "examples": self.examples,
            "background": self.background,
            "communication_style": self.communication_style,
            "key_principles": self.key_principles
        }


class PersonaContextLoader:
    """Loads and manages persona configurations."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the persona context loader.
        
        Args:
            config_dir: Directory containing persona configuration files.
                       Defaults to config/personas/
        """
        if config_dir is None:
            # Default to config/personas/ relative to project root
            self.config_dir = Path(__file__).parent.parent / "config" / "personas"
        else:
            self.config_dir = Path(config_dir)
        
        self._persona_cache: Dict[str, PersonaContext] = {}
        self._expert_type_map: Dict[str, str] = {}
        
    def load_persona(self, persona_name: str) -> PersonaContext:
        """
        Load a persona configuration by name.
        
        Args:
            persona_name: Name of the persona to load
            
        Returns:
            PersonaContext object with loaded configuration
            
        Raises:
            FileNotFoundError: If persona configuration file doesn't exist
            ValueError: If persona configuration is invalid
        """
        # Check cache first
        if persona_name in self._persona_cache:
            return self._persona_cache[persona_name]
        
        # Try to find persona file
        persona_file = self._find_persona_file(persona_name)
        
        if not persona_file.exists():
            raise FileNotFoundError(
                f"Persona configuration not found for '{persona_name}'. "
                f"Searched in: {self.config_dir}"
            )
        
        # Load YAML configuration
        with open(persona_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not config:
            raise ValueError(f"Empty or invalid persona configuration in {persona_file}")
        
        # Create PersonaContext
        persona = PersonaContext(
            name=config.get('name', persona_name),
            role=config.get('role', ''),
            expertise=config.get('expertise', []),
            tone=config.get('tone', 'professional'),
            perspective=config.get('perspective', 'objective'),
            constraints=config.get('constraints', []),
            examples=config.get('examples', []),
            background=config.get('background', ''),
            communication_style=config.get('communication_style', ''),
            key_principles=config.get('key_principles', [])
        )
        
        # Cache the persona
        self._persona_cache[persona_name] = persona
        
        # Map expert types to personas
        if 'expert_types' in config:
            for expert_type in config['expert_types']:
                self._expert_type_map[expert_type] = persona_name
        
        return persona
    
    def _find_persona_file(self, persona_name: str) -> Path:
        """
        Find the persona configuration file for a given persona name.
        
        Args:
            persona_name: Name of the persona
            
        Returns:
            Path to the persona configuration file
        """
        # Try exact match first
        exact_file = self.config_dir / f"{persona_name}.yaml"
        if exact_file.exists():
            return exact_file
        
        # Try lowercase version
        lowercase_file = self.config_dir / f"{persona_name.lower()}.yaml"
        if lowercase_file.exists():
            return lowercase_file
        
        # Try underscore version
        underscore_file = self.config_dir / f"{persona_name.replace(' ', '_').lower()}.yaml"
        if underscore_file.exists():
            return underscore_file
        
        # Return the exact match path (will fail if it doesn't exist)
        return exact_file
    
    def get_persona_for_expert_type(self, expert_type: str) -> PersonaContext:
        """
        Get the persona context for a specific expert type.
        
        Args:
            expert_type: Type of expert (e.g., 'python_reviewer', 'sql_reviewer')
            
        Returns:
            PersonaContext for the expert type
            
        Raises:
            ValueError: If no persona is configured for the expert type
        """
        # Check if we have a mapping
        if expert_type in self._expert_type_map:
            persona_name = self._expert_type_map[expert_type]
            return self.load_persona(persona_name)
        
        # Try to load persona directly by expert type name
        try:
            return self.load_persona(expert_type)
        except FileNotFoundError:
            raise ValueError(
                f"No persona configuration found for expert type '{expert_type}'. "
                f"Available expert types: {list(self._expert_type_map.keys())}"
            )
    
    def list_available_personas(self) -> List[str]:
        """
        List all available persona configurations.
        
        Returns:
            List of persona names
        """
        if not self.config_dir.exists():
            return []
        
        personas = []
        for file in self.config_dir.glob("*.yaml"):
            personas.append(file.stem)
        
        return sorted(personas)
    
    def list_expert_types(self) -> List[str]:
        """
        List all expert types that have persona mappings.
        
        Returns:
            List of expert type names
        """
        return sorted(self._expert_type_map.keys())
    
    def clear_cache(self):
        """Clear the persona cache."""
        self._persona_cache.clear()
        self._expert_type_map.clear()


def build_persona_prompt_section(persona: PersonaContext) -> str:
    """
    Build a formatted prompt section from persona context.
    
    Args:
        persona: PersonaContext object
        
    Returns:
        Formatted string suitable for inclusion in a prompt
    """
    sections = []
    
    # Name and Role
    sections.append(f"## Persona: {persona.name}")
    sections.append(f"**Role:** {persona.role}\n")
    
    # Background (if available)
    if persona.background:
        sections.append(f"**Background:** {persona.background}\n")
    
    # Expertise
    if persona.expertise:
        sections.append("**Areas of Expertise:**")
        for area in persona.expertise:
            sections.append(f"- {area}")
        sections.append("")
    
    # Communication Style (if available)
    if persona.communication_style:
        sections.append(f"**Communication Style:** {persona.communication_style}\n")
    
    # Tone
    sections.append(f"**Tone:** {persona.tone}\n")
    
    # Perspective
    sections.append(f"**Perspective:** {persona.perspective}\n")
    
    # Key Principles (if available)
    if persona.key_principles:
        sections.append("**Key Principles:**")
        for principle in persona.key_principles:
            sections.append(f"- {principle}")
        sections.append("")
    
    # Constraints
    if persona.constraints:
        sections.append("**Constraints & Guidelines:**")
        for constraint in persona.constraints:
            sections.append(f"- {constraint}")
        sections.append("")
    
    # Examples
    if persona.examples:
        sections.append("**Example Responses:**")
        for i, example in enumerate(persona.examples, 1):
            sections.append(f"{i}. {example}")
        sections.append("")
    
    return "\n".join(sections)


def get_default_persona_context(expert_type: str) -> PersonaContext:
    """
    Get a default persona context for an expert type when no configuration exists.
    
    Args:
        expert_type: Type of expert
        
    Returns:
        Default PersonaContext
    """
    # Extract a readable name from expert_type
    name_parts = expert_type.replace('_', ' ').title().split()
    readable_name = ' '.join(name_parts)
    
    return PersonaContext(
        name=readable_name,
        role=f"Expert {readable_name}",
        expertise=[expert_type.replace('_', ' ').title()],
        tone="professional",
        perspective="objective",
        constraints=[
            "Provide accurate and helpful responses",
            "Stay within your area of expertise",
            "Be clear and concise"
        ],
        examples=[],
        background=f"You are an expert in {expert_type.replace('_', ' ')}.",
        communication_style="Clear, professional, and informative",
        key_principles=[
            "Accuracy",
            "Clarity",
            "Relevance"
        ]
    )


# Convenience function for quick persona loading
def get_persona_for_expert(expert_type: str, config_dir: Optional[str] = None) -> PersonaContext:
    """
    Convenience function to load a persona for an expert type.
    
    Args:
        expert_type: Type of expert
        config_dir: Optional directory containing persona configurations
        
    Returns:
        PersonaContext for the expert type
    """
    loader = PersonaContextLoader(config_dir)
    try:
        return loader.get_persona_for_expert_type(expert_type)
    except ValueError:
        # Fall back to default persona
        return get_default_persona_context(expert_type)
