"""
Test cases for classify_query function (Option A fixes)

Tests the conservative fixes applied to prevent misclassification:
1. HTML/CSS pattern requires action verbs
2. Python pattern requires action verbs
3. Trivial Q&A word limit reduced to 6
4. Expert validation with fallback
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_v13 import classify_query, QueryTier, SAVANT_MODELS


def test_classify_query():
    """Test all classification scenarios"""
    
    test_cases = [
        # TIER 1: Trivial calculations
        ("4+5", QueryTier.TIER1_TRIVIAL, None, "simple_calculation", True),
        ("what is 10-3", QueryTier.TIER1_TRIVIAL, None, "simple_calculation", True),
        ("calculate 5*6", QueryTier.TIER1_TRIVIAL, None, "simple_calculation", True),
        
        # TIER 1: Short Q&A (word_count <= 6)
        ("what is python?", QueryTier.TIER1_TRIVIAL, None, "short_qa", True),
        ("define recursion", QueryTier.TIER1_TRIVIAL, None, "short_qa", True),
        ("who is Turing", QueryTier.TIER1_TRIVIAL, None, "short_qa", True),
        
        # Edge case: Multi-word question (doesn't match single-word pattern)
        # Pattern r'^what\s+is\s+\w+\??$' only matches "what is [single_word]"
        ("what is the meaning of life", QueryTier.TIER3_COMPLEX, None, "complex", False),
        
        # NOT TIER 1: Too many words (>6)
        ("what is the detailed process of photosynthesis", QueryTier.TIER3_COMPLEX, None, "complex", False),
        
        # TIER 2: Math problems
        ("solve this equation x+5=10", QueryTier.TIER2_SINGLE, "math_expert", "math", False),
        ("find the derivative of x^2", QueryTier.TIER2_SINGLE, "math_expert", "math", False),
        
        # TIER 2: SQL queries
        ("select * from users", QueryTier.TIER2_SINGLE, "sql_schema_architect", "sql", False),
        ("write a sql query", QueryTier.TIER2_SINGLE, "sql_schema_architect", "sql", False),
        
        # TIER 2: HTML/CSS with action verbs (FIXED)
        ("create an HTML page", QueryTier.TIER2_SINGLE, "html_css_specialist", "html", False),
        ("build a webpage with CSS", QueryTier.TIER2_SINGLE, "html_css_specialist", "html", False),
        ("design a landing page", QueryTier.TIER2_SINGLE, "html_css_specialist", "html", False),
        
        # NOT TIER 2: HTML without action verbs (FIXED - should be TIER3)
        ("what is HTML?", QueryTier.TIER1_TRIVIAL, None, "short_qa", True),  # Caught by trivial Q&A
        ("explain CSS flexbox", QueryTier.TIER3_COMPLEX, None, "complex", False),
        ("HTML is a markup language", QueryTier.TIER3_COMPLEX, None, "complex", False),
        
        # TIER 2: Python with action verbs (FIXED)
        ("write a python script to parse JSON", QueryTier.TIER2_SINGLE, "python_backend", "python", False),
        ("create a python function for sorting", QueryTier.TIER2_SINGLE, "python_backend", "python", False),
        
        # NOT TIER 2: Python without action verbs (FIXED)
        ("python is a programming language", QueryTier.TIER3_COMPLEX, None, "complex", False),
        
        # NOT TIER 2: Python with web/api keywords (should escalate to TIER3)
        ("write a python flask web app", QueryTier.TIER3_COMPLEX, None, "complex", False),
        ("create a python api with fastapi", QueryTier.TIER3_COMPLEX, None, "complex", False),
        
        # TIER 3: Complex multi-expert queries
        ("Build a web app with Python Flask and React", QueryTier.TIER3_COMPLEX, None, "complex", False),
        ("Create a full-stack application", QueryTier.TIER3_COMPLEX, None, "complex", False),
    ]
    
    print("Testing classify_query with Option A fixes...\n")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for query, expected_tier, expected_expert, expected_reason, expected_suppress in test_cases:
        tier, expert_id, title, metadata = classify_query(query)
        
        # Check tier
        tier_match = tier == expected_tier
        
        # Check expert (only for TIER2)
        expert_match = True
        if expected_tier == QueryTier.TIER2_SINGLE:
            expert_match = expert_id == expected_expert
        
        # Check reason
        reason_match = metadata['reason'] == expected_reason or metadata['reason'].startswith(expected_reason)
        
        # Check suppress_thinking
        suppress_match = metadata.get('suppress_thinking', False) == expected_suppress
        
        # Overall pass/fail
        test_passed = tier_match and expert_match and reason_match and suppress_match
        
        if test_passed:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
            
        print(f"{status} | {query[:50]:50s}")
        
        if not test_passed:
            print(f"       Expected: tier={expected_tier.name}, expert={expected_expert}, reason={expected_reason}, suppress={expected_suppress}")
            print(f"       Got:      tier={tier.name}, expert={expert_id}, reason={metadata['reason']}, suppress={metadata.get('suppress_thinking', False)}")
            print()
    
    print("=" * 80)
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = test_classify_query()
    sys.exit(0 if success else 1)
