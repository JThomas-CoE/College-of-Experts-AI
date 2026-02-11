"""
Output Structure Enforcer - Validates mandatory output structure
College of Experts V12 Architecture

Moved from demo_v12_full.py to break circular import with src/quality_gate.py
"""

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class StructuredOutput:
    """Parsed output with outline and sections."""
    raw_content: str
    has_outline: bool
    outline: List[str]
    sections: Dict[str, str]
    is_valid: bool
    validation_errors: List[str]


class OutputStructureEnforcer:
    """
    Enforces mandatory output structure:
    1. Outline section at the top
    2. Numbered hierarchical sections (1, 1.1, 1.1.1, etc.)
    """
    
    def validate(self, output: str) -> StructuredOutput:
        """Check if output follows mandatory structure."""
        errors = []
        
        # Check for outline
        has_outline = bool(re.search(r'##?\s*Outline', output, re.IGNORECASE))
        if not has_outline:
            has_outline = bool(re.search(r'^\d+\.\s+\[', output, re.MULTILINE))
        
        if not has_outline:
            errors.append("Missing outline section")
        
        # Extract outline items
        outline = []
        outline_match = re.search(r'##?\s*Outline\s*([\s\S]*?)(?=\n##|\n---|\Z)', output, re.IGNORECASE)
        if outline_match:
            outline_text = outline_match.group(1)
            outline = re.findall(r'^\s*\d+\..*$', outline_text, re.MULTILINE)
        
        # Check for numbered sections
        sections = {}
        section_pattern = r'##?\s*(\d+(?:\.\d+)*)[.\s]+([^\n]+)\n([\s\S]*?)(?=\n##?\s*\d+|\Z)'
        for match in re.finditer(section_pattern, output):
            section_num = match.group(1)
            section_content = match.group(3).strip()
            sections[section_num] = section_content
        
        if not sections:
            errors.append("No numbered sections found")
        
        # Check for code in coding tasks
        has_code = bool(re.search(r'```python|def\s+\w+|class\s+\w+', output))
        
        return StructuredOutput(
            raw_content=output,
            has_outline=has_outline,
            outline=outline,
            sections=sections,
            is_valid=len(errors) == 0 or has_code,
            validation_errors=errors
        )
