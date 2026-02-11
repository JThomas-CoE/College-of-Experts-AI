"""
Router Model - Handles scope negotiation and expert dispatch.

⚠️  DEPRECATION WARNING ⚠️
This module is DEPRECATED as of 2026-02-01 and is no longer used in demo_v13.py.

MIGRATION PATH:
- Use `MemoryVectorRouter` from `src.memory_router` for memory-aware routing
- Use `VectorRouter` from `src.vector_router` for FAISS-based similarity search
- See src/deprecated/README.md for details

This file is kept for backward compatibility with:
- demo.py (old demo)
- tests/test_router.py (legacy tests)

The router is the always-resident coordinator that:
1. Classifies user intent and scope
2. Selects appropriate expert(s) for the task
3. Handles lightweight queries directly
4. Detects transitions requiring expert changes
"""

import json
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

try:
    import ollama
except ImportError:
    ollama = None


@dataclass
class ExpertRecommendation:
    """Recommendation from router about which experts to load."""
    expert_ids: list[str]
    confidence: float
    reasoning: str
    needs_clarification: bool
    clarification_question: Optional[str] = None


@dataclass
class RouterConfig:
    """Configuration for the router model."""
    model_name: str = "qwen3:4b"  # Default router model
    temperature: float = 0.3
    max_tokens: int = 512
    manifest_path: Optional[Path] = None


class Router:
    """
    The router handles scope negotiation and expert selection.
    
    It stays resident in memory and coordinates loading/unloading
    of specialized experts based on task requirements.
    """
    
    SYSTEM_PROMPT = """You are a routing assistant for the College of Experts system.
Your job is to analyze user requests and determine which specialized experts are needed.

Available experts:
{expert_list}

For each user request, you must:
1. Classify the domain(s) involved
2. Determine if you need clarification to scope the task
3. Recommend which expert(s) to load

Respond in JSON format:
{{
    "needs_clarification": true/false,
    "clarification_question": "question if needed, else null",
    "recommended_experts": ["expert_id1", "expert_id2"],
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

If the query is trivial (definitions, conversions, simple facts), respond:
{{
    "handle_directly": true,
    "response": "your direct answer"
}}
"""

    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()
        self.expert_manifest = self._load_expert_manifest()
        
        if ollama is None:
            raise ImportError("ollama package required. Install with: pip install ollama")
    
    def _load_expert_manifest(self) -> dict:
        """Load the manifest of available experts."""
        if self.config.manifest_path and self.config.manifest_path.exists():
            with open(self.config.manifest_path) as f:
                return json.load(f)
        
        # Default manifest for demo
        return {
            "code_python": {
                "display_name": "Python Expert",
                "model": "deepseek-coder-v3:33b",
                "domains": ["python", "programming", "debugging"],
                "description": "Specialized in Python development"
            },
            "code_general": {
                "display_name": "General Coding Expert", 
                "model": "qwen3-coder:14b",
                "domains": ["programming", "code", "software"],
                "description": "General software development"
            },
            "math": {
                "display_name": "Mathematics Expert",
                "model": "qwen3-math:7b",
                "domains": ["math", "calculations", "proofs"],
                "description": "Mathematical reasoning and problem solving"
            },
            "writing": {
                "display_name": "Writing Expert",
                "model": "llama4:8b",
                "domains": ["writing", "editing", "creative"],
                "description": "Creative and technical writing"
            },
            "general": {
                "display_name": "General Assistant",
                "model": "qwen3:7b",
                "domains": ["general", "research", "analysis"],
                "description": "General knowledge and analysis"
            }
        }
    
    def _format_expert_list(self) -> str:
        """Format expert manifest for system prompt."""
        lines = []
        for expert_id, info in self.expert_manifest.items():
            domains = ", ".join(info.get("domains", []))
            lines.append(f"- {expert_id}: {info['display_name']} ({domains})")
        return "\n".join(lines)
    
    def classify(self, user_input: str, context: Optional[str] = None) -> ExpertRecommendation:
        """
        Classify user input and recommend experts.
        
        Args:
            user_input: The user's message
            context: Optional conversation context
            
        Returns:
            ExpertRecommendation with expert IDs and metadata
        """
        system_prompt = self.SYSTEM_PROMPT.format(
            expert_list=self._format_expert_list()
        )
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if context:
            messages.append({"role": "user", "content": f"Context: {context}"})
        
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = ollama.chat(
                model=self.config.model_name,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            )
            
            result = self._parse_response(response["message"]["content"])
            return result
            
        except Exception as e:
            # Fallback to general expert on error
            return ExpertRecommendation(
                expert_ids=["general"],
                confidence=0.5,
                reasoning=f"Fallback due to error: {str(e)}",
                needs_clarification=False
            )
    
    def _parse_response(self, response_text: str) -> ExpertRecommendation:
        """Parse the JSON response from the router model."""
        try:
            # Try to extract JSON from response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                data = json.loads(json_str)
                
                # Check if router wants to handle directly
                if data.get("handle_directly"):
                    return ExpertRecommendation(
                        expert_ids=[],  # Empty = router handles it
                        confidence=1.0,
                        reasoning="Router handling directly",
                        needs_clarification=False
                    )
                
                return ExpertRecommendation(
                    expert_ids=data.get("recommended_experts", ["general"]),
                    confidence=data.get("confidence", 0.7),
                    reasoning=data.get("reasoning", ""),
                    needs_clarification=data.get("needs_clarification", False),
                    clarification_question=data.get("clarification_question")
                )
        except json.JSONDecodeError:
            pass
        
        # Fallback parsing
        return ExpertRecommendation(
            expert_ids=["general"],
            confidence=0.5,
            reasoning="Could not parse router response",
            needs_clarification=False
        )
    
    def handle_trivial(self, user_input: str) -> Optional[str]:
        """
        Attempt to handle trivial queries directly without loading experts.
        
        Returns response string if handled, None if experts needed.
        """
        messages = [
            {
                "role": "system", 
                "content": """Answer simple factual questions directly and briefly.
If the question requires specialized expertise or extended reasoning, respond with:
{"needs_expert": true}"""
            },
            {"role": "user", "content": user_input}
        ]
        
        try:
            response = ollama.chat(
                model=self.config.model_name,
                messages=messages,
                options={"temperature": 0.1, "num_predict": 256}
            )
            
            content = response["message"]["content"]
            if '{"needs_expert": true}' in content or '"needs_expert"' in content:
                return None
            return content
            
        except Exception:
            return None


if __name__ == "__main__":
    # Quick test
    router = Router()
    
    test_queries = [
        "Help me write a Python script to parse JSON",
        "What is 2 + 2?",
        "Review my React code for security issues",
        "Help me with my project"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = router.classify(query)
        print(f"  Experts: {result.expert_ids}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Needs clarification: {result.needs_clarification}")
        if result.clarification_question:
            print(f"  Question: {result.clarification_question}")
