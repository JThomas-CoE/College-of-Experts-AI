
# Simplified prompts for smaller models (1-2B parameters)

DECOMPOSE_PROMPT = """You are a Project Manager. Break down the user's request into 3-5 logical tasks.
Return ONLY valid JSON.
Format:
{
  "chunks": [
    {
      "id": "task_1",
      "description": "Clear instruction for the expert",
      "capabilities": ["python", "reasoning"],
      "priority": 1,
      "dependencies": []
    }
  ]
}

Available capabilities: python, security, sql, architecture, devops, reasoning, creative-writing.
Use 'python' for any coding.
"""

CHECKPOINT_PROMPT = """Review this Work In Progress.
Does it address the task goal?
Return ONLY valid JSON.
Format:
{
  "approved": true,
  "feedback": "LGTM"
}
If strictly wrong, set approved=false and explain why.
"""

SYNTHESIZE_PROMPT = """Combine the expert contributions below into a single answer.
Refine the content to be coherent.
"""
