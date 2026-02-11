"""
Assist Resolver - Resolves {{ASSIST:type:domain:description}} placeholders
"""

import re
from typing import List, Dict, Tuple, Optional


class AssistResolver:
    """Resolves {{ASSIST:type:domain:description}} placeholders and filters duplicate code blocks."""
    
    ASSIST_PATTERN = r'\{\{ASSIST:(\w+):([^:}]+):([^}]+)\}\}'
    CODE_BLOCK_PATTERN = r'```(\w+)?\n(.*?)```'  # Non-greedy match for code blocks
    
    def _deduplicate_code_blocks(self, content: str) -> str:
        """
        Keep only the last occurrence of each code block language.
        
        Models sometimes output placeholder templates first, then the actual
        styled content. This removes earlier placeholders and keeps the final version.
        
        Args:
            content: Text containing potentially duplicate code blocks
            
        Returns:
            Content with only last occurrence of each language kept
        """
        import re
        
        # Find all code blocks with their positions
        blocks = []
        for match in re.finditer(r'```(\w+)?\n(.*?)```', content, re.DOTALL):
            lang = match.group(1) or 'text'
            code = match.group(2)
            blocks.append({
                'lang': lang.lower(),
                'full_match': match.group(0),
                'code': code,
                'start': match.start(),
                'end': match.end()
            })
        
        if len(blocks) <= 1:
            return content  # No duplicates to worry about
        
        # Group by language and keep only last occurrence of each
        last_by_lang = {}
        for block in blocks:
            last_by_lang[block['lang']] = block
        
        # If we have duplicate languages, reconstruct content keeping only last of each
        if len(last_by_lang) < len(blocks):
            # Sort blocks by position
            blocks.sort(key=lambda x: x['start'])
            
            # Build result keeping only last of each language
            result = content
            kept_indices = set()
            for lang, block in last_by_lang.items():
                # Find this block's index in the sorted list
                for i, b in enumerate(blocks):
                    if b['start'] == block['start']:
                        kept_indices.add(i)
                        break
            
            # Remove all code blocks then add back only the kept ones
            # This is a simplified approach - remove all then re-insert
            # Actually, let's just keep the last occurrence of duplicates
            result_parts = []
            last_end = 0
            seen_langs = set()
            
            # Process blocks in reverse to identify which to keep
            for block in reversed(blocks):
                if block['lang'] not in seen_langs:
                    seen_langs.add(block['lang'])
                    # Mark as keep
                    block['keep'] = True
                else:
                    block['keep'] = False
            
            # Rebuild content
            for block in blocks:
                if block['keep']:
                    result_parts.append(content[last_end:block['start']])
                    result_parts.append(block['full_match'])
                    last_end = block['end']
            
            result_parts.append(content[last_end:])
            return ''.join(result_parts)
        
        return content  # No duplicates found
    
    def resolve_content(self, content: str, mode: str = "autonomous") -> Tuple[str, List[Dict]]:
        """
        Resolve placeholders AND deduplicate code blocks in content.
        
        Args:
            content: Text containing assist placeholders and/or duplicate code blocks
            mode: "autonomous" (best effort) or "interactive" (ask user)
            
        Returns:
            Tuple of (resolved_content, unresolved_placeholders)
        """
        # First deduplicate code blocks
        content = self._deduplicate_code_blocks(content)
        
        # Then resolve placeholders
        return self.resolve(content, mode)
    
    def extract_placeholders(self, content: str) -> List[Dict]:
        """Extract all assist placeholders from content."""
        matches = re.findall(self.ASSIST_PATTERN, content)
        return [{"type": m[0], "domain": m[1], "description": m[2]} for m in matches]
    
    def resolve(self, content: str, mode: str = "autonomous") -> Tuple[str, List[Dict]]:
        """
        Resolve placeholders in content.
        
        Args:
            content: Text containing assist placeholders
            mode: "autonomous" (best effort) or "interactive" (ask user)
            
        Returns:
            Tuple of (resolved_content, unresolved_placeholders)
        """
        placeholders = self.extract_placeholders(content)
        unresolved = []
        
        for ph in placeholders:
            placeholder_str = f"{{{{ASSIST:{ph['type']}:{ph['domain']}:{ph['description']}}}}}"
            
            if ph['type'] == 'lookup':
                resolution = self._heuristic_lookup(ph['domain'], ph['description'])
                if resolution:
                    content = content.replace(placeholder_str, resolution)
                else:
                    unresolved.append(ph)
            else:
                unresolved.append(ph)
        
        return content, unresolved
    
    def _heuristic_lookup(self, domain: str, description: str) -> Optional[str]:
        """Provide heuristic resolutions for common lookups."""
        desc_lower = description.lower()
        
        # Session timeouts
        if "session timeout" in desc_lower:
            return "30 minutes (OWASP recommendation)"
        
        # Password hashing
        if "password hash" in desc_lower or "bcrypt" in desc_lower:
            return "bcrypt with cost factor 12"
        
        # JWT
        if "jwt expiry" in desc_lower or "token expiry" in desc_lower:
            return "15 minutes for access tokens, 7 days for refresh tokens"
        
        # Encryption
        if "encryption" in desc_lower and "aes" in desc_lower:
            return "AES-256-GCM"
        
        # TLS
        if "tls" in desc_lower or "ssl" in desc_lower:
            return "TLS 1.3"
        
        # Database ports
        if "port" in desc_lower and "postgres" in desc_lower:
            return "5432"
        if "port" in desc_lower and "mysql" in desc_lower:
            return "3306"
        
        return None
