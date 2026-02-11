"""
DeepSeek R1 Output Parser

Handles parsing of DeepSeek R1 model outputs including:
- Stripping <think>...</think> reasoning blocks
- Extracting JSON from code blocks
- Extracting rationale text

DeepSeek R1 uses thinking mode where it outputs reasoning in <think> tags
before the final answer. This parser cleanly separates thinking from output.
"""

import re
import json
import logging
from typing import Tuple, Optional, Any, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedResponse:
    """Parsed DeepSeek R1 response."""
    thinking: Optional[str]  # Content from <think>...</think> block
    rationale: Optional[str]  # Extracted rationale text
    json_data: Optional[Dict[str, Any]]  # Parsed JSON if found
    raw_output: str  # Output after thinking stripped
    success: bool  # Whether JSON was successfully extracted
    error: Optional[str]  # Error message if parsing failed


def strip_thinking(text: str) -> Tuple[str, Optional[str]]:
    """
    Remove <think>...</think> blocks from DeepSeek R1 output.
    
    Args:
        text: Raw model output potentially containing thinking blocks
        
    Returns:
        Tuple of (cleaned_text, thinking_content)
        thinking_content is None if no thinking block found
    """
    # Pattern matches <think>...</think> including newlines
    pattern = r'<think>(.*?)</think>'
    
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    
    if match:
        thinking_content = match.group(1).strip()
        cleaned = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE).strip()
        return cleaned, thinking_content
    
    return text.strip(), None


def extract_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Extract JSON from text, looking for code blocks first, then raw JSON.
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Tuple of (parsed_json, error_message)
        parsed_json is None if extraction failed
    """
    # Try to find JSON in code block first (```json ... ```)
    json_block_pattern = r'```json\s*(.*?)\s*```'
    match = re.search(json_block_pattern, text, flags=re.DOTALL)
    
    if match:
        json_str = match.group(1).strip()
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, (dict, list)):
                return parsed, None
        except json.JSONDecodeError as e:
            logger.warning(f"JSON in code block is malformed: {e}")
            # Continue to try other methods
    
    # Try generic code block (``` ... ```)
    generic_block_pattern = r'```\s*(.*?)\s*```'
    match = re.search(generic_block_pattern, text, flags=re.DOTALL)
    
    if match:
        potential_json = match.group(1).strip()
        # Check if it looks like JSON (starts with { or [)
        if potential_json.startswith('{') or potential_json.startswith('['):
            try:
                parsed = json.loads(potential_json)
                if isinstance(parsed, (dict, list)):
                    return parsed, None
            except json.JSONDecodeError:
                pass  # Continue to try other methods
    
    # Try to find raw JSON object
    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(brace_pattern, text, flags=re.DOTALL)
    
    # Try each match, starting with the longest (most likely to be complete)
    for potential_json in sorted(matches, key=len, reverse=True):
        try:
            parsed = json.loads(potential_json)
            if isinstance(parsed, (dict, list)):
                return parsed, None
        except json.JSONDecodeError:
            continue
    
    # Try to find the largest balanced JSON structure
    json_obj = find_balanced_json(text)
    if json_obj:
        try:
            parsed = json.loads(json_obj)
            if isinstance(parsed, (dict, list)):
                return parsed, None
            return None, f"Parsed JSON is not a structured object (got {type(parsed)})"
        except json.JSONDecodeError as e:
            return None, f"Found JSON-like structure but failed to parse: {e}"
    
    return None, "No valid JSON found in output"


def find_balanced_json(text: str) -> Optional[str]:
    """
    Find the largest balanced JSON object in text.
    
    Args:
        text: Text to search
        
    Returns:
        JSON string if found, None otherwise
    """
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    depth = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\' and in_string:
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if in_string:
            continue
            
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start_idx:i+1]
    
    return None


def extract_rationale(text: str) -> Optional[str]:
    """
    Extract rationale text from output.
    
    Looks for patterns like:
    - RATIONALE: ...
    - Rationale: ...
    - ## Rationale ...
    
    Args:
        text: Text to search
        
    Returns:
        Rationale text if found, None otherwise
    """
    patterns = [
        r'RATIONALE:\s*(.*?)(?=\n\n|\nADAPTED|\n```|$)',
        r'Rationale:\s*(.*?)(?=\n\n|\nAdapted|\n```|$)',
        r'##\s*Rationale\s*(.*?)(?=\n##|\n```|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


def parse_deepseek_response(raw_output: str) -> ParsedResponse:
    """
    Parse a complete DeepSeek R1 response.
    
    Handles:
    1. Stripping <think>...</think> blocks
    2. Extracting rationale text
    3. Extracting and parsing JSON
    
    Args:
        raw_output: Raw model output
        
    Returns:
        ParsedResponse with all extracted components
    """
    # Step 1: Strip thinking blocks
    cleaned, thinking = strip_thinking(raw_output)
    
    # Step 2: Extract rationale
    rationale = extract_rationale(cleaned)
    
    # Step 3: Extract JSON
    json_data, error = extract_json(cleaned)
    
    return ParsedResponse(
        thinking=thinking,
        rationale=rationale,
        json_data=json_data,
        raw_output=cleaned,
        success=json_data is not None,
        error=error
    )


def format_thinking_for_debug(thinking: Optional[str], max_length: int = 500) -> str:
    """
    Format thinking content for debug logging.
    
    Args:
        thinking: Raw thinking content
        max_length: Maximum length to include
        
    Returns:
        Formatted string for logging
    """
    if not thinking:
        return "[No thinking block]"
    
    if len(thinking) <= max_length:
        return thinking
    
    return thinking[:max_length] + f"... [{len(thinking) - max_length} chars truncated]"


# Convenience functions for common use cases

def parse_json_only(raw_output: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Quick JSON extraction, stripping thinking first.
    
    Args:
        raw_output: Raw model output
        
    Returns:
        Tuple of (json_data, error_message)
    """
    cleaned, _ = strip_thinking(raw_output)
    return extract_json(cleaned)


def parse_with_rationale(raw_output: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    Extract JSON and rationale from output.
    
    Args:
        raw_output: Raw model output
        
    Returns:
        Tuple of (json_data, rationale, error_message)
    """
    result = parse_deepseek_response(raw_output)
    return result.json_data, result.rationale, result.error


if __name__ == "__main__":
    # Test the parser with sample output
    sample_output = """<think>
Let me analyze this query about creating a FastAPI authentication system.

The user needs:
1. JWT-based authentication
2. User registration and login
3. Protected routes

I should adapt the fastapi_auth_jwt template but add a registration slot.
</think>

RATIONALE:
Selected fastapi_auth_jwt template. Added user registration slot since the query 
explicitly mentions "user registration". Removed the refresh_token slot as it 
wasn't mentioned in requirements.

ADAPTED FRAMEWORK:
```json
{
    "id": "adapted_fastapi_auth_20260127",
    "title": "FastAPI JWT Authentication with Registration",
    "description": "Authentication system with JWT tokens and user registration",
    "slots": [
        {"id": "user_model", "title": "User Model", "dependencies": []},
        {"id": "registration", "title": "User Registration", "dependencies": ["user_model"]},
        {"id": "auth_utils", "title": "JWT Utilities", "dependencies": []},
        {"id": "login_route", "title": "Login Endpoint", "dependencies": ["user_model", "auth_utils"]},
        {"id": "protected_routes", "title": "Protected Routes", "dependencies": ["auth_utils"]}
    ]
}
```
"""
    
    result = parse_deepseek_response(sample_output)
    
    print("=" * 60)
    print("DeepSeek Parser Test")
    print("=" * 60)
    print(f"\nSuccess: {result.success}")
    print(f"\nThinking (truncated): {format_thinking_for_debug(result.thinking, 200)}")
    print(f"\nRationale: {result.rationale}")
    print(f"\nJSON Data: {json.dumps(result.json_data, indent=2) if result.json_data else 'None'}")
    if result.error:
        print(f"\nError: {result.error}")
