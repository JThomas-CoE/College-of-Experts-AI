"""
V12 Structured Output - Outline Enforcement
College of Experts Architecture

Handles:
- Mandatory outline → sections output structure
- Output validation and parsing
- Template guidance for experts
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


class OutputStructureError(Exception):
    """Raised when output doesn't match required structure."""
    pass


@dataclass
class OutlineItem:
    """Single item in an outline."""
    number: str          # "1", "1.1", "2", etc.
    title: str
    level: int           # Depth level (1, 2, 3...)
    description: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "number": self.number,
            "title": self.title,
            "level": self.level,
            "description": self.description
        }


@dataclass
class Section:
    """Parsed section from structured output."""
    number: str
    title: str
    content: str
    level: int = 1
    subsections: List["Section"] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "number": self.number,
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "subsections": [s.to_dict() for s in self.subsections]
        }
    
    @property
    def full_text(self) -> str:
        """Get full text including subsections."""
        parts = [self.content]
        for sub in self.subsections:
            parts.append(sub.full_text)
        return "\n\n".join(parts)


@dataclass
class ParsedStructuredOutput:
    """Result of parsing structured expert output."""
    outline: List[OutlineItem]
    sections: Dict[str, Section]
    raw_text: str
    validation_warnings: List[str] = field(default_factory=list)
    
    def get_section(self, number: str) -> Optional[Section]:
        """Get section by number (e.g., '1.1')."""
        return self.sections.get(number)
    
    def get_outline_text(self) -> str:
        """Get outline as formatted text."""
        lines = []
        for item in self.outline:
            indent = "  " * (item.level - 1)
            lines.append(f"{indent}{item.number}. {item.title}")
        return "\n".join(lines)
    
    def get_summary(self, max_chars: int = 500) -> str:
        """Get summary of the output."""
        parts = []
        for item in self.outline[:5]:
            parts.append(f"{item.number}. {item.title}")
        
        summary = " | ".join(parts)
        if len(self.outline) > 5:
            summary += f" | ... (+{len(self.outline) - 5} more)"
        
        return summary[:max_chars]


class OutlineEnforcer:
    """
    Enforces and validates mandatory outline → sections structure.
    """
    
    # Regex patterns for parsing
    OUTLINE_HEADER_PATTERN = re.compile(
        r'^#+\s*outline:?\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    
    SECTION_NUMBER_PATTERN = re.compile(
        r'^(\d+(?:\.\d+)*)[.)\s]+(.+)$'
    )
    
    SECTION_HEADER_PATTERN = re.compile(
        r'^#+\s*(\d+(?:\.\d+)*)[.)\s]+(.+)$',
        re.MULTILINE
    )
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize enforcer.
        
        Args:
            strict_mode: If True, raise errors on structure violations.
                        If False, log warnings and continue.
        """
        self.strict_mode = strict_mode
    
    def generate_template(
        self,
        slot_title: str,
        slot_description: str,
        expected_outputs: List[str],
        suggested_outline: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate output template/guidance for an expert.
        
        Args:
            slot_title: Title of the task slot
            slot_description: Description of required work
            expected_outputs: List of expected output items
            suggested_outline: Optional suggested outline structure
            
        Returns:
            Template string for expert guidance
        """
        template_parts = [
            f"## Task: {slot_title}\n",
            f"{slot_description}\n",
            "\n---\n",
            "### Required Output Structure\n",
            "Your response MUST follow this exact structure:\n",
            "\n**1. Outline Section** (required first)\n",
            "```\n## Outline\n",
            "1. <First Major Section>\n",
            "   1.1 <Subsection if needed>\n",
            "2. <Second Major Section>\n",
            "...\n```\n",
            "\n**2. Numbered Sections** (following outline)\n",
            "Each outline item must have a corresponding section:\n",
            "```\n## 1. <First Major Section>\n",
            "<content here>\n",
            "### 1.1 <Subsection>\n",
            "<content here>\n```\n",
            "\n---\n",
            "### Expected Outputs\n"
        ]
        
        for i, output in enumerate(expected_outputs, 1):
            template_parts.append(f"- {output}\n")
        
        if suggested_outline:
            template_parts.append("\n### Suggested Outline\n")
            for item in suggested_outline:
                indent = "  " * (item.get("level", 1) - 1)
                template_parts.append(f"{indent}{item['number']}. {item['title']}\n")
        
        template_parts.extend([
            "\n---\n",
            "### Important Notes\n",
            "- Start with `## Outline` section\n",
            "- Use numbered sections matching the outline (1., 1.1, 2., etc.)\n",
            "- Do not skip section numbers\n",
            "- Include all expected outputs in appropriate sections\n",
            "- Use markdown headers (## for main, ### for sub, #### for sub-sub)\n"
        ])
        
        return "".join(template_parts)
    
    def parse_output(self, text: str) -> ParsedStructuredOutput:
        """
        Parse structured output into outline and sections.
        
        Args:
            text: Raw text output from expert
            
        Returns:
            ParsedStructuredOutput with parsed structure
            
        Raises:
            OutputStructureError if strict_mode and structure invalid
        """
        warnings = []
        
        # Find outline section
        outline, outline_end = self._extract_outline(text)
        
        if not outline:
            msg = "No outline section found (expected '## Outline')"
            if self.strict_mode:
                raise OutputStructureError(msg)
            warnings.append(msg)
            outline = []
        
        # Parse numbered sections
        sections = self._extract_sections(text, outline_end)
        
        # Validate outline matches sections
        validation_errors = self._validate_structure(outline, sections)
        warnings.extend(validation_errors)
        
        if self.strict_mode and validation_errors:
            raise OutputStructureError(
                f"Structure validation failed: {validation_errors}"
            )
        
        return ParsedStructuredOutput(
            outline=outline,
            sections=sections,
            raw_text=text,
            validation_warnings=warnings
        )
    
    def _extract_outline(self, text: str) -> Tuple[List[OutlineItem], int]:
        """
        Extract outline from text.
        
        Returns:
            Tuple of (outline items, position where outline ends)
        """
        outline_items = []
        outline_end = 0
        
        # Find outline header
        match = self.OUTLINE_HEADER_PATTERN.search(text)
        if not match:
            return [], 0
        
        start_pos = match.end()
        
        # Find next major header (indicating outline end)
        next_header = re.search(r'\n##\s+\d', text[start_pos:])
        if next_header:
            outline_end = start_pos + next_header.start()
            outline_text = text[start_pos:outline_end]
        else:
            outline_text = text[start_pos:]
            outline_end = len(text)
        
        # Parse outline items
        for line in outline_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Count leading spaces for level
            original_line = line
            level = 1
            
            # Match numbered item
            match = self.SECTION_NUMBER_PATTERN.match(line)
            if match:
                number = match.group(1)
                title = match.group(2).strip()
                
                # Calculate level from number (1.2.3 = level 3)
                level = number.count('.') + 1
                
                outline_items.append(OutlineItem(
                    number=number,
                    title=title,
                    level=level
                ))
        
        return outline_items, outline_end
    
    def _extract_sections(
        self,
        text: str,
        start_pos: int = 0
    ) -> Dict[str, Section]:
        """
        Extract numbered sections from text.
        
        Args:
            text: Full text
            start_pos: Position to start searching
            
        Returns:
            Dict mapping section numbers to Section objects
        """
        sections = {}
        section_text = text[start_pos:]
        
        # Find all section headers
        headers = list(self.SECTION_HEADER_PATTERN.finditer(section_text))
        
        for i, match in enumerate(headers):
            number = match.group(1)
            title = match.group(2).strip()
            
            # Content is everything between this header and next
            content_start = match.end()
            if i < len(headers) - 1:
                content_end = headers[i + 1].start()
            else:
                content_end = len(section_text)
            
            content = section_text[content_start:content_end].strip()
            
            # Calculate level from number
            level = number.count('.') + 1
            
            sections[number] = Section(
                number=number,
                title=title,
                content=content,
                level=level
            )
        
        # Build hierarchy (attach subsections to parents)
        self._build_section_hierarchy(sections)
        
        return sections
    
    def _build_section_hierarchy(self, sections: Dict[str, Section]):
        """Attach subsections to their parent sections."""
        for number, section in sections.items():
            if '.' in number:
                # Find parent
                parent_number = number.rsplit('.', 1)[0]
                if parent_number in sections:
                    sections[parent_number].subsections.append(section)
    
    def _validate_structure(
        self,
        outline: List[OutlineItem],
        sections: Dict[str, Section]
    ) -> List[str]:
        """
        Validate that outline matches sections.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check each outline item has a section
        for item in outline:
            if item.number not in sections:
                errors.append(
                    f"Outline item '{item.number}. {item.title}' "
                    f"has no corresponding section"
                )
        
        # Check for orphan sections (sections not in outline)
        outline_numbers = {item.number for item in outline}
        for number in sections:
            if number not in outline_numbers:
                errors.append(
                    f"Section '{number}' not in outline"
                )
        
        return errors
    
    def format_sections_for_reference(
        self,
        output: ParsedStructuredOutput,
        requested_sections: Optional[List[str]] = None,
        include_outline: bool = True
    ) -> str:
        """
        Format parsed output for use as reference by other slots.
        
        Args:
            output: Parsed structured output
            requested_sections: Specific section numbers to include (None = all)
            include_outline: Whether to include outline
            
        Returns:
            Formatted reference text
        """
        parts = []
        
        if include_outline:
            parts.append("### Reference Outline")
            parts.append(output.get_outline_text())
            parts.append("")
        
        parts.append("### Reference Sections")
        
        sections_to_include = requested_sections or list(output.sections.keys())
        
        for number in sections_to_include:
            section = output.get_section(number)
            if section:
                parts.append(f"\n#### {number}. {section.title}")
                parts.append(section.content)
        
        return "\n".join(parts)


# Instruction templates for expert prompting
STRUCTURED_OUTPUT_INSTRUCTIONS = """
## Output Structure Requirements

Your response MUST follow this mandatory structure:

### 1. Outline (Required First)
Start with a numbered outline of your response:
```
## Outline
1. First Major Section
   1.1 Subsection Name
   1.2 Another Subsection
2. Second Major Section
3. Third Major Section
```

### 2. Numbered Sections
After the outline, provide content for each item:
```
## 1. First Major Section
<content>

### 1.1 Subsection Name
<detailed content>

### 1.2 Another Subsection
<detailed content>

## 2. Second Major Section
<content>
```

### Rules:
- ALWAYS start with `## Outline`
- Use ## for main sections, ### for subsections
- Number format: 1, 1.1, 1.2, 2, 2.1, etc.
- Every outline item MUST have a corresponding section
- Do not skip numbers
"""


def create_expert_prompt(
    task_description: str,
    expected_outputs: List[str],
    reference_context: Optional[str] = None,
    suggested_outline: Optional[List[Dict]] = None
) -> str:
    """
    Create a complete expert prompt with structure requirements.
    
    Args:
        task_description: Description of the task
        expected_outputs: List of expected output items
        reference_context: Optional context from other slots
        suggested_outline: Optional suggested outline structure
        
    Returns:
        Complete prompt string
    """
    parts = [
        "# Expert Task\n",
        f"{task_description}\n",
        "\n---\n",
        "## Expected Outputs\n"
    ]
    
    for output in expected_outputs:
        parts.append(f"- {output}\n")
    
    if reference_context:
        parts.extend([
            "\n---\n",
            "## Reference Context\n",
            "You may reference the following completed work:\n\n",
            reference_context,
            "\n"
        ])
    
    parts.extend([
        "\n---\n",
        STRUCTURED_OUTPUT_INSTRUCTIONS
    ])
    
    if suggested_outline:
        parts.extend([
            "\n---\n",
            "## Suggested Outline\n",
            "Consider using this structure:\n"
        ])
        for item in suggested_outline:
            indent = "  " * (item.get("level", 1) - 1)
            parts.append(f"{indent}{item['number']}. {item['title']}\n")
    
    parts.extend([
        "\n---\n",
        "Please provide your response following the structure above."
    ])
    
    return "".join(parts)


if __name__ == "__main__":
    # Test the structured output parser
    print("Testing OutlineEnforcer...")
    
    # Example structured output
    test_output = """
## Outline
1. Security Architecture
   1.1 Encryption Approach
   1.2 Access Control Model
2. Implementation Details
3. Testing Strategy

## 1. Security Architecture
This section covers the overall security design for HIPAA compliance.

### 1.1 Encryption Approach
- AES-256 for data at rest
- TLS 1.3 for data in transit
- Key rotation every 90 days

### 1.2 Access Control Model
- Role-based access control (RBAC)
- Minimum necessary principle
- Audit logging for all access

## 2. Implementation Details
The implementation uses Python with SQLAlchemy for database access.
Key components:
- Connection pooling
- Prepared statements
- Transaction management

## 3. Testing Strategy
Testing includes:
- Unit tests for all functions
- Integration tests with mock database
- Security penetration testing
"""
    
    enforcer = OutlineEnforcer(strict_mode=False)
    
    # Parse the output
    result = enforcer.parse_output(test_output)
    
    print(f"\nOutline items: {len(result.outline)}")
    for item in result.outline:
        indent = "  " * (item.level - 1)
        print(f"  {indent}{item.number}. {item.title}")
    
    print(f"\nSections parsed: {len(result.sections)}")
    for number, section in result.sections.items():
        print(f"  {number}. {section.title} ({len(section.content)} chars)")
    
    print(f"\nValidation warnings: {result.validation_warnings}")
    
    print("\n--- Formatted for reference ---")
    ref_text = enforcer.format_sections_for_reference(
        result,
        requested_sections=["1.1", "2"],
        include_outline=True
    )
    print(ref_text[:500] + "...")
    
    print("\n--- Expert prompt example ---")
    prompt = create_expert_prompt(
        task_description="Implement HIPAA-compliant database access in Python",
        expected_outputs=["Connection code", "Encryption functions", "Error handling"],
        suggested_outline=[
            {"number": "1", "title": "Setup", "level": 1},
            {"number": "2", "title": "Implementation", "level": 1},
            {"number": "2.1", "title": "Encryption", "level": 2}
        ]
    )
    print(prompt[:800] + "...")
