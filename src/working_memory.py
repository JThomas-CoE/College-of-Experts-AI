"""
V12 Working Memory - RAM-based Slot Storage
College of Experts Architecture

Handles:
- Storage of completed framework slot outputs
- Hierarchical reference retrieval (outline, sections, full)
- Read-only access for subsequent slots
- Memory separate from KV cache
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class OutlineItem:
    """Represents an item in a structured outline."""
    number: str          # e.g., "1", "1.1", "1.1.2"
    title: str           # Section title
    level: int           # Nesting depth (1, 2, 3...)
    children: List["OutlineItem"] = field(default_factory=list)


@dataclass
class StructuredSlotResult:
    """
    Parsed result from an expert with hierarchical structure.
    Enables efficient section-level reference retrieval.
    """
    slot_id: str
    expert_id: str
    raw_content: str
    outline: List[OutlineItem]
    sections: Dict[str, str]      # "1.1" → section content
    section_titles: Dict[str, str]  # "1.1" → section title
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def parse(cls, slot_id: str, expert_id: str, raw_content: str) -> "StructuredSlotResult":
        """
        Parse raw expert output into structured result.
        
        Expected format:
        ## Outline
        1. [Title]
           1.1 [Subtitle]
        2. [Title]
        
        ---
        
        ## 1. Title
        [Content...]
        
        ### 1.1 Subtitle
        [Content...]
        """
        outline = cls._extract_outline(raw_content)
        sections, section_titles = cls._extract_sections(raw_content)
        
        return cls(
            slot_id=slot_id,
            expert_id=expert_id,
            raw_content=raw_content,
            outline=outline,
            sections=sections,
            section_titles=section_titles
        )
    
    @staticmethod
    def _extract_outline(content: str) -> List[OutlineItem]:
        """Extract outline items from content."""
        outline = []
        
        # Find outline section
        outline_match = re.search(
            r'##\s*Outline\s*\n(.*?)(?=\n---|\n##\s+\d)',
            content,
            re.DOTALL | re.IGNORECASE
        )
        
        if not outline_match:
            return outline
        
        outline_text = outline_match.group(1)
        
        # Parse numbered items
        # Matches: "1. Title" or "1.1 Title" or "   1.2.3 Title"
        pattern = r'^(\s*)(\d+(?:\.\d+)*)[.\s]+(.+)$'
        
        for line in outline_text.split('\n'):
            match = re.match(pattern, line)
            if match:
                indent = len(match.group(1))
                number = match.group(2)
                title = match.group(3).strip()
                level = number.count('.') + 1
                
                outline.append(OutlineItem(
                    number=number,
                    title=title,
                    level=level
                ))
        
        return outline
    
    @staticmethod
    def _extract_sections(content: str) -> tuple:
        """Extract numbered sections from content."""
        sections = {}
        section_titles = {}
        
        # Pattern for section headers: ## 1. Title or ### 1.1 Subtitle
        pattern = r'^(#{2,4})\s*(\d+(?:\.\d+)*)[.\s]+(.+)$'
        
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            match = re.match(pattern, line)
            if match:
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = match.group(2)
                section_titles[current_section] = match.group(3).strip()
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections, section_titles
    
    def get_outline_only(self) -> str:
        """Return formatted outline without content."""
        if not self.outline:
            return "[No outline available]"
        
        lines = ["## Outline"]
        for item in self.outline:
            indent = "  " * (item.level - 1)
            lines.append(f"{indent}{item.number}. {item.title}")
        
        return '\n'.join(lines)
    
    def get_section(self, label: str) -> Optional[str]:
        """
        Get specific section by number or fuzzy title match.
        
        Args:
            label: Section number (e.g., "1.1") or partial title
            
        Returns:
            Section content or None if not found
        """
        # Try exact number match
        if label in self.sections:
            title = self.section_titles.get(label, "")
            return f"### {label}. {title}\n\n{self.sections[label]}"
        
        # Try fuzzy title match
        label_lower = label.lower()
        for num, title in self.section_titles.items():
            if label_lower in title.lower():
                return f"### {num}. {title}\n\n{self.sections[num]}"
        
        return None
    
    def get_sections(self, labels: List[str]) -> str:
        """Get multiple sections concatenated."""
        results = []
        for label in labels:
            section = self.get_section(label)
            if section:
                results.append(section)
        
        return '\n\n---\n\n'.join(results) if results else "[Sections not found]"
    
    def get_summary(self, max_chars: int = 500) -> str:
        """Get brief summary: outline + truncated first section."""
        summary_parts = [self.get_outline_only()]
        
        if self.sections:
            first_section_num = min(self.sections.keys())
            first_content = self.sections[first_section_num]
            if len(first_content) > max_chars:
                first_content = first_content[:max_chars] + "..."
            summary_parts.append(f"\n---\n\n### Preview of {first_section_num}:\n{first_content}")
        
        return '\n'.join(summary_parts)


class WorkingMemory:
    """
    RAM-based storage for completed framework slot outputs.
    
    Key characteristics:
    - Stored in RAM, NOT in KV cache
    - Read-only access for subsequent slots
    - Hierarchical reference retrieval
    - Injected into prompts only when explicitly referenced
    """
    
    def __init__(self):
        self._completed: Dict[str, StructuredSlotResult] = {}
        self._access_log: List[Dict] = []
    
    def store(self, slot_id: str, expert_id: str, raw_content: str) -> StructuredSlotResult:
        """
        Store completed slot output.
        
        Args:
            slot_id: Unique slot identifier
            expert_id: Expert that produced the output
            raw_content: Raw expert output text
            
        Returns:
            Parsed StructuredSlotResult
        """
        result = StructuredSlotResult.parse(slot_id, expert_id, raw_content)
        self._completed[slot_id] = result
        return result
    
    def store_parsed(self, result: StructuredSlotResult):
        """Store an already-parsed result."""
        self._completed[result.slot_id] = result
    
    def has(self, slot_id: str) -> bool:
        """Check if slot output exists."""
        return slot_id in self._completed
    
    def get(self, slot_id: str) -> Optional[StructuredSlotResult]:
        """Get full structured result for a slot."""
        return self._completed.get(slot_id)
    
    def get_reference(
        self,
        slot_id: str,
        reference_type: str = "full",
        section_hints: Optional[List[str]] = None,
        max_tokens: int = 2000
    ) -> str:
        """
        Get reference content for a slot.
        
        Args:
            slot_id: Slot to reference
            reference_type: "full", "outline", "sections", or "summary"
            section_hints: Specific sections to retrieve (for "sections" type)
            max_tokens: Maximum approximate token count
            
        Returns:
            Reference content string
        """
        if slot_id not in self._completed:
            return f"[Reference not available: {slot_id}]"
        
        result = self._completed[slot_id]
        
        # Log access
        self._access_log.append({
            "slot_id": slot_id,
            "reference_type": reference_type,
            "section_hints": section_hints,
            "timestamp": datetime.now().isoformat()
        })
        
        if reference_type == "outline":
            return result.get_outline_only()
        
        elif reference_type == "sections" and section_hints:
            return result.get_sections(section_hints)
        
        elif reference_type == "summary":
            return result.get_summary(max_chars=max_tokens * 4)  # ~4 chars per token
        
        else:  # "full"
            content = result.raw_content
            if len(content) > max_tokens * 4:
                # Truncate with notice
                content = content[:max_tokens * 4]
                content += f"\n\n[... truncated, {len(result.raw_content)} chars total ...]"
            return content
    
    def build_reference_context(
        self,
        references: Dict[str, Dict],
        max_total_tokens: int = 4000
    ) -> str:
        """
        Build reference context for a slot from multiple dependencies.
        
        Args:
            references: Dict mapping slot_id to reference config
                       e.g., {"slot_a": {"type": "sections", "hints": ["1.1", "2"]}}
            max_total_tokens: Maximum total tokens for all references
            
        Returns:
            Formatted reference context string
        """
        if not references:
            return ""
        
        parts = []
        tokens_per_ref = max_total_tokens // len(references)
        
        for slot_id, config in references.items():
            ref_type = config.get("type", "outline")
            hints = config.get("hints", [])
            
            content = self.get_reference(
                slot_id,
                reference_type=ref_type,
                section_hints=hints,
                max_tokens=tokens_per_ref
            )
            
            parts.append(f"### Reference: {slot_id}\n\n{content}")
        
        return '\n\n---\n\n'.join(parts)
    
    def get_all_completed(self) -> List[str]:
        """Get list of all completed slot IDs."""
        return list(self._completed.keys())
    
    def get_access_log(self) -> List[Dict]:
        """Get log of all reference accesses."""
        return self._access_log.copy()
    
    def clear(self):
        """Clear all stored results."""
        self._completed.clear()
        self._access_log.clear()
    
    def get_signatures(self, slot_id: str) -> str:
        """
        Get published signatures/interfaces for a completed slot.
        This is the READ-ONLY interface for downstream slots.
        
        Returns minimal interface info - class names, function signatures, constants.
        This keeps downstream KV caches small.
        """
        if slot_id not in self._completed:
            return f"[Slot {slot_id} not found in memory]"
        
        result = self._completed[slot_id]
        
        # Check if signatures already extracted in metadata
        if "signatures" in result.metadata:
            return result.metadata["signatures"]
        
        # Extract signatures on-demand
        signatures = self._extract_signatures(result.raw_content)
        result.metadata["signatures"] = signatures
        return signatures
    
    def get_all_signatures(self, slot_ids: List[str]) -> str:
        """Get signatures for multiple slots - for downstream reference."""
        parts = []
        for slot_id in slot_ids:
            if self.has(slot_id):
                sigs = self.get_signatures(slot_id)
                parts.append(f"# From {slot_id}:\n{sigs}")
        return "\n\n".join(parts) if parts else ""
    
    @staticmethod
    def _extract_signatures(raw_content: str) -> str:
        """
        Extract class/function signatures from code output.
        Publishes minimal interface info for downstream slots.
        """
        signatures = []
        
        # Extract class definitions with inheritance
        for match in re.finditer(r'^class\s+(\w+)(?:\([^)]*\))?:', raw_content, re.MULTILINE):
            signatures.append(match.group(0).rstrip(':'))
        
        # Extract function/method definitions with params
        for match in re.finditer(r'^(?:async\s+)?def\s+(\w+)\s*\([^)]*\)(?:\s*->\s*[^:]+)?:', raw_content, re.MULTILINE):
            sig = match.group(0).rstrip(':')
            signatures.append(sig)
        
        # Extract important constants (UPPER_CASE = ...)
        for match in re.finditer(r'^([A-Z][A-Z0-9_]*)\s*=\s*(.{1,50})', raw_content, re.MULTILINE):
            const_name = match.group(1)
            const_preview = match.group(2).split('\n')[0][:30]
            signatures.append(f"{const_name} = {const_preview}...")
        
        # Extract router/app instances
        for match in re.finditer(r'^(\w+)\s*=\s*(APIRouter|FastAPI|Flask)\s*\(', raw_content, re.MULTILINE):
            signatures.append(f"{match.group(1)} = {match.group(2)}()")
        
        if not signatures:
            return "[No code signatures found]"
        
        return "\n".join(signatures)


# Utility function for parsing without full WorkingMemory
def parse_expert_output(raw_content: str) -> Dict:
    """
    Quick parse of expert output to check structure.
    
    Returns:
        Dict with has_outline, has_sections, section_count, issues
    """
    result = {
        "has_outline": False,
        "has_sections": False,
        "section_count": 0,
        "issues": []
    }
    
    # Check for outline
    if re.search(r'##\s*Outline', raw_content, re.IGNORECASE):
        result["has_outline"] = True
    else:
        result["issues"].append("Missing '## Outline' section")
    
    # Check for numbered sections
    section_pattern = r'^#{2,4}\s*\d+(?:\.\d+)*[.\s]+'
    sections = re.findall(section_pattern, raw_content, re.MULTILINE)
    result["section_count"] = len(sections)
    result["has_sections"] = len(sections) > 0
    
    if not result["has_sections"]:
        result["issues"].append("Missing numbered section headers (## 1. Title)")
    
    return result


if __name__ == "__main__":
    # Test working memory
    print("Testing WorkingMemory...")
    
    # Sample expert output
    sample_output = """## Outline
1. Security Requirements
   1.1 Authentication
   1.2 Encryption
2. Implementation Approach
3. Testing Strategy

---

## 1. Security Requirements

This section covers the core security requirements for the system.

### 1.1 Authentication

Users must authenticate using OAuth 2.0 with JWT tokens.
- Token expiration: 1 hour
- Refresh token rotation enabled

### 1.2 Encryption

All data must be encrypted:
- At rest: AES-256-GCM
- In transit: TLS 1.3

## 2. Implementation Approach

We will implement using a layered architecture:
1. API Gateway for authentication
2. Service layer for business logic
3. Data layer with encrypted storage

## 3. Testing Strategy

Testing will include:
- Unit tests for encryption functions
- Integration tests for auth flow
- Penetration testing by third party
"""
    
    # Initialize working memory
    memory = WorkingMemory()
    
    # Store result
    result = memory.store("security_design", "security_architect", sample_output)
    
    print("\n--- Outline Only ---")
    print(result.get_outline_only())
    
    print("\n--- Section 1.1 ---")
    print(result.get_section("1.1"))
    
    print("\n--- Fuzzy Match 'encryption' ---")
    print(result.get_section("encryption"))
    
    print("\n--- Summary ---")
    print(result.get_summary(max_chars=300))
    
    print("\n--- Reference Context ---")
    context = memory.build_reference_context({
        "security_design": {"type": "sections", "hints": ["1.1", "1.2"]}
    })
    print(context)
    
    print("\n--- Parse Check ---")
    check = parse_expert_output(sample_output)
    print(f"Has outline: {check['has_outline']}")
    print(f"Has sections: {check['has_sections']}")
    print(f"Section count: {check['section_count']}")
    print(f"Issues: {check['issues']}")
