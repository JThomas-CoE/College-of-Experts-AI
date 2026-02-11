"""
Tests for the Router module.
"""

import pytest
from unittest.mock import patch, MagicMock

# Import will fail without ollama, so we mock it
import sys
sys.modules['ollama'] = MagicMock()

from src.router import Router, RouterConfig, ExpertRecommendation


class TestRouter:
    """Tests for the Router class."""
    
    def test_router_creation(self):
        """Test that router can be created with default config."""
        router = Router()
        assert router.config.model_name == "qwen2.5:3b"
        assert len(router.expert_manifest) > 0
    
    def test_router_custom_config(self):
        """Test router with custom configuration."""
        config = RouterConfig(
            model_name="custom:model",
            temperature=0.5
        )
        router = Router(config)
        assert router.config.model_name == "custom:model"
        assert router.config.temperature == 0.5
    
    def test_expert_manifest_loaded(self):
        """Test that default expert manifest is loaded."""
        router = Router()
        assert "code_python" in router.expert_manifest
        assert "math" in router.expert_manifest
        assert "general" in router.expert_manifest
    
    def test_format_expert_list(self):
        """Test expert list formatting for prompt."""
        router = Router()
        expert_list = router._format_expert_list()
        assert "code_python" in expert_list
        assert "Python Expert" in expert_list
    
    def test_parse_response_valid_json(self):
        """Test parsing valid JSON responses."""
        router = Router()
        
        response = '''Here's my analysis:
        {
            "needs_clarification": false,
            "recommended_experts": ["code_python", "math"],
            "confidence": 0.9,
            "reasoning": "User needs Python and math help"
        }'''
        
        result = router._parse_response(response)
        
        assert isinstance(result, ExpertRecommendation)
        assert result.expert_ids == ["code_python", "math"]
        assert result.confidence == 0.9
        assert result.needs_clarification is False
    
    def test_parse_response_clarification_needed(self):
        """Test parsing response that needs clarification."""
        router = Router()
        
        response = '''{
            "needs_clarification": true,
            "clarification_question": "What programming language?",
            "recommended_experts": ["code_general"],
            "confidence": 0.5
        }'''
        
        result = router._parse_response(response)
        
        assert result.needs_clarification is True
        assert result.clarification_question == "What programming language?"
    
    def test_parse_response_invalid_json(self):
        """Test fallback when JSON parsing fails."""
        router = Router()
        
        response = "This is not valid JSON at all"
        result = router._parse_response(response)
        
        assert result.expert_ids == ["general"]
        assert result.confidence == 0.5
    
    def test_parse_response_handle_directly(self):
        """Test parsing direct-handling response."""
        router = Router()
        
        response = '''{
            "handle_directly": true,
            "response": "The answer is 42"
        }'''
        
        result = router._parse_response(response)
        
        assert result.expert_ids == []  # Empty means router handles it
        assert result.confidence == 1.0


class TestExpertRecommendation:
    """Tests for ExpertRecommendation dataclass."""
    
    def test_recommendation_creation(self):
        """Test creating an expert recommendation."""
        rec = ExpertRecommendation(
            expert_ids=["code_python"],
            confidence=0.95,
            reasoning="Python code detected",
            needs_clarification=False
        )
        
        assert rec.expert_ids == ["code_python"]
        assert rec.confidence == 0.95
        assert rec.clarification_question is None
    
    def test_recommendation_with_clarification(self):
        """Test recommendation with clarification question."""
        rec = ExpertRecommendation(
            expert_ids=["general"],
            confidence=0.3,
            reasoning="Unclear request",
            needs_clarification=True,
            clarification_question="Can you be more specific?"
        )
        
        assert rec.needs_clarification is True
        assert rec.clarification_question == "Can you be more specific?"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
