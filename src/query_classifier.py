"""
Query Classifier - LLM-Based Complexity Assessment

Uses DeepSeek to intelligently classify query complexity instead of brittle regex patterns.

Architecture:
- Stage 1: Fast-path regex for trivial queries (< 1ms)
- Stage 2: LLM assessment for everything else (2-5 seconds)

This replaces the old regex-based classify_query() function with intelligent
model-based classification that properly handles multi-domain queries.
"""

import re
import json
import logging
from enum import Enum
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# [Architecture Compliance] Use centralized robust parser
from src.deepseek_parser import extract_json, strip_thinking


class QueryTier(Enum):
    """Query complexity tiers for adaptive processing."""
    TIER1_TRIVIAL = 1  # Direct answer with thinking suppressed
    TIER2_SINGLE = 2   # Single expert routing
    TIER3_COMPLEX = 3  # Full multi-expert decomposition


@dataclass
class ClassificationResult:
    """Result of query classification."""
    tier: QueryTier
    experts_needed: List[str]
    deliverables_count: int
    reasoning: str
    metadata: Dict[str, Any]
    
    @property
    def primary_expert(self) -> Optional[str]:
        """Get the primary expert for TIER2 single-expert routing."""
        return self.experts_needed[0] if self.experts_needed else None


# LLM Classification Prompt
CLASSIFICATION_PROMPT = """You are a query complexity analyzer. Analyze this query and determine how to process it.

QUERY:
{query}

AVAILABLE EXPERTS:
- python_backend: Python scripts, Flask/FastAPI backends, general coding
- sql_schema_architect: Database schemas, SQL queries, data modeling
- html_css_specialist: Frontend HTML/CSS/JavaScript, web UI
- security_architect: Authentication, encryption, security audits
- math_expert: Calculations, equations, mathematical proofs
- legal_contracts: Legal contracts and agreements

CLASSIFICATION RULES:
1. TIER1_TRIVIAL: Simple factual Q&A, basic arithmetic, OR follow-up questions about code (e.g., "how do I run this?", "explain the backend")
2. TIER2_SINGLE: Task requiring ONE expert domain (e.g., "write a Python sort function")
3. TIER3_COMPLEX: Tasks requiring MULTIPLE experts OR multiple distinct deliverables

KEY INDICATORS OF TIER3_COMPLEX:
- Numbered lists of requirements (1. 2. 3.)
- Multiple file types needed (HTML + Python + SQL)
- Words like "full-stack", "complete app", "with database and frontend"
- Security + Backend + Frontend combined

RESPOND IN VALID JSON ONLY (no markdown, no explanation outside JSON):
{
  "tier": "TIER1_TRIVIAL" | "TIER2_SINGLE" | "TIER3_COMPLEX",
  "experts_needed": ["expert_id1", "expert_id2"],
  "deliverables_count": 1,
  "reasoning": "one sentence explanation"
}"""


class QueryClassifier:
    """
    Intelligent query classifier using LLM assessment.
    
    Two-stage classification:
    1. Fast-path: Regex for obvious trivial queries (sub-millisecond)
    2. LLM assessment: DeepSeek evaluates complexity (2-5 seconds)
    """
    
    # Available expert IDs for validation
    VALID_EXPERTS = {
        "python_backend", "sql_schema_architect", "html_css_specialist",
        "security_architect", "math_expert", "legal_contracts"
    }
    
    def __init__(self, executor: Any, deepseek_slot: Optional[str] = None, router: Optional[Any] = None):
        """
        Initialize classifier.
        
        Args:
            executor: ModelExecutor instance for LLM calls
            deepseek_slot: Pre-loaded DeepSeek slot ID (optional, will load if needed)
            router: MemoryVectorRouter/DualRouter instance for fallback routing
        """
        self.executor = executor
        self.deepseek_slot = deepseek_slot
        self.router = router
    
    def set_deepseek_slot(self, slot_id: str) -> None:
        """Update the DeepSeek slot ID after loading."""
        self.deepseek_slot = slot_id
    
    async def classify(self, query: str) -> ClassificationResult:
        """
        Classify query complexity.
        
        Args:
            query: User query to classify
            
        Returns:
            ClassificationResult with tier, experts, and metadata
        """
        # Stage 1: Fast-path for trivial queries
        trivial_result = self._check_trivial(query)
        if trivial_result:
            return trivial_result
        
        # Stage 2: LLM-based classification
        try:
            return await self._llm_classify(query)
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            
            # 1. Trivial Check (Short query fallback)
            if len(query.split()) < 10:
                 return ClassificationResult(
                    tier=QueryTier.TIER1_TRIVIAL,
                    experts_needed=[],
                    deliverables_count=0,
                    reasoning="Fallback: Short query with LLM failure",
                    metadata={"fallback": True}
                )

            # 2. Structural Fallback: Delegate to Semantic Router
            # Uses embeddings to find the best expert instead of guessing keywords
            if self.router:
                try:
                    route_result = self.router.route(query)
                    expert = route_result.get("expert_id", "python_backend")
                    
                    return ClassificationResult(
                        tier=QueryTier.TIER2_SINGLE,
                        experts_needed=[expert],
                        deliverables_count=1,
                        reasoning=f"Fallback: Semantic Router delegation after error {e}",
                        metadata={"error": str(e), "fallback": True}
                    )
                except Exception as re:
                     logger.warning(f"Router fallback failed: {re}")

            return ClassificationResult(
                tier=QueryTier.TIER3_COMPLEX,
                experts_needed=[],
                deliverables_count=1,
                reasoning=f"Classification error, defaulting to complex: {e}",
                metadata={"error": str(e), "fallback": True}
            )
    
    def _check_trivial(self, query: str) -> Optional[ClassificationResult]:
        """
        Fast-path check for trivial queries.
        
        Only checks for obviously trivial patterns that don't need LLM assessment.
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        word_count = len(words)
        
        # Simple arithmetic: "4+5", "what is 10-3"
        calc_patterns = [
            r'^\d+\s*[\+\-\*\/x×]\s*\d+\s*\??$',
            r'^what\s+is\s+\d+[\+\-\*\/]',
            r'^calculate\s+\d+[\+\-\*\/]'
        ]
        for pattern in calc_patterns:
            if re.match(pattern, query_lower):
                return ClassificationResult(
                    tier=QueryTier.TIER1_TRIVIAL,
                    experts_needed=[],
                    deliverables_count=0,
                    reasoning="Simple arithmetic calculation",
                    metadata={"suppress_thinking": True, "fast_path": "calculation"}
                )
        
        # Very short factual Q&A (≤4 words)
        short_qa_patterns = [
            r'^what\s+is\s+\w+\??$',
            r'^define\s+\w+\??$',
            r'^who\s+is\s+\w+\??$'
        ]
        for pattern in short_qa_patterns:
            if re.match(pattern, query_lower) and word_count <= 4:
                return ClassificationResult(
                    tier=QueryTier.TIER1_TRIVIAL,
                    experts_needed=[],
                    deliverables_count=0,
                    reasoning="Short factual question",
                    metadata={"suppress_thinking": True, "fast_path": "short_qa"}
                )
        
        # Not trivial - needs LLM assessment
        return None
    
    async def _llm_classify(self, query: str) -> ClassificationResult:
        """
        Use LLM to assess query complexity.
        
        Calls DeepSeek with a structured prompt and parses JSON response.
        """
        prompt = CLASSIFICATION_PROMPT.replace("{query}", query)
        
        # Generate classification
        response = await self.executor.generate(
            prompt,
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=256,
            slot_id=self.deepseek_slot
        )
        
        # Parse JSON response
        result = self._parse_response(response, query)
        return result
    
    def _parse_response(self, response: str, original_query: str) -> ClassificationResult:
        """
        Parse LLM classification response using robust deepseek_parser.
        
        Handles:
        - <think> tags (stripping)
        - Markdown code blocks
        - Raw JSON (missing braces)
        - Key-value pairs
        """
        try:
            # Step 1: Strip thinking tokens using standard parser
            cleaned_text, _ = strip_thinking(response)
            
            # Step 2: Extract JSON using robust extractor (handles blocks, braces, etc.)
            data, error_msg = extract_json(cleaned_text)
            
            if not data:
                # If standard extraction failed, logging raw response for debug
                # The robust parser already tries: code blocks, raw json, balanced braces
                raise ValueError(f"No valid JSON found: {error_msg}")
            
            # Map tier string to enum
            tier_str = data.get("tier", "TIER3_COMPLEX")
            tier_map = {
                "TIER1_TRIVIAL": QueryTier.TIER1_TRIVIAL,
                "TIER2_SINGLE": QueryTier.TIER2_SINGLE,
                "TIER3_COMPLEX": QueryTier.TIER3_COMPLEX
            }
            tier = tier_map.get(tier_str, QueryTier.TIER3_COMPLEX)
            
            # Validate and filter experts
            experts = data.get("experts_needed", [])
            valid_experts = [e for e in experts if e in self.VALID_EXPERTS]
            
            # If TIER2 but no valid expert, default to python_backend
            if tier == QueryTier.TIER2_SINGLE and not valid_experts:
                valid_experts = ["python_backend"]

            # [Logic Fix] Force TIER 2 if simple structure (1 expert, 1 deliverable)
            # This aligns with user expectation: "Single item query is simple or Tier 2"
            if len(valid_experts) == 1 and data.get("deliverables_count", 1) == 1:
                 if tier == QueryTier.TIER3_COMPLEX:
                      tier = QueryTier.TIER2_SINGLE
                      # Ensure expert is set if not already
                      pass
            
            return ClassificationResult(
                tier=tier,
                experts_needed=valid_experts,
                deliverables_count=data.get("deliverables_count", 1),
                reasoning=data.get("reasoning", "LLM classification"),
                metadata={"raw_response": response[:500], "parsed": True}
            )
            
        except (ValueError, KeyError, TypeError, Exception) as e:
            # Fallback 1: Text Regex (If JSON fails but model thought correctly)
            import re
            tier_match = re.search(r'(TIER[123]_[A-Z]+)', response)
            if tier_match:
                tier_str = tier_match.group(1)
                tier_map = {
                    "TIER1_TRIVIAL": QueryTier.TIER1_TRIVIAL,
                    "TIER2_SINGLE": QueryTier.TIER2_SINGLE,
                    "TIER3_COMPLEX": QueryTier.TIER3_COMPLEX
                }
                found_tier = tier_map.get(tier_str)
                
                if found_tier:
                    # If we found a tier but no JSON, delegate expert selection to Router
                    expert = ["python_backend"]
                    if self.router:
                         try:
                            # Use Semantic Router to find the expert
                            route = self.router.route(original_query)
                            expert = [route.get("expert_id", "python_backend")]
                         except:
                            pass

                    return ClassificationResult(
                        tier=found_tier,
                        experts_needed=expert,
                        deliverables_count=1,
                        reasoning=f"Recovered tier from text: {tier_str}",
                        metadata={"parsed": False, "recovered": True}
                    )
            
            # Fallback 2: Heuristics
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._fallback_classify(original_query, response)
    
    def _fallback_classify(self, query: str, raw_response: str) -> ClassificationResult:
        """
        Fallback classification when LLM response parsing fails.
        
        Uses simple heuristics to make a reasonable guess.
        """
        query_lower = query.lower()
        
        # Check for complexity indicators
        numbered_items = len(re.findall(r'^\s*\d+[\.\)]\s+', query, re.MULTILINE))
        
        domain_keywords = ['html', 'css', 'python', 'flask', 'database', 'sql', 
                          'api', 'backend', 'frontend', 'authentication']
        domain_hits = sum(1 for kw in domain_keywords if kw in query_lower)
        
        if numbered_items >= 2 or domain_hits >= 3:
            return ClassificationResult(
                tier=QueryTier.TIER3_COMPLEX,
                experts_needed=[],
                deliverables_count=max(numbered_items, 2),
                reasoning="Fallback: detected multi-part or multi-domain query",
                metadata={"fallback": True, "raw_response": raw_response[:200]}
            )
        
        # Default to single expert
        return ClassificationResult(
            tier=QueryTier.TIER2_SINGLE,
            experts_needed=["python_backend"],
            deliverables_count=1,
            reasoning="Fallback: defaulting to single expert",
            metadata={"fallback": True}
        )


# Convenience function for backward compatibility
def create_classifier(executor: Any, deepseek_slot: Optional[str] = None) -> QueryClassifier:
    """Create a QueryClassifier instance."""
    return QueryClassifier(executor, deepseek_slot)
