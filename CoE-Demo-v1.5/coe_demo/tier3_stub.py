
import re
import json
import time
import os
import sys
import argparse
import tempfile
from typing import *
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import ast
import urllib.request

from .config import *
from .models import *


def run_tier3(
    query: str,
    domains: List[str],
    mgr: "ModelManager",
    session_store: "SessionStore" = None,
    template_store: "TemplateStore" = None
) -> str:
    print("\n[TIER3] Complex Multi-Domain Request Detected", file=sys.stderr)
    breakdown_prompt = f"""The user has submitted a complex query requiring multiple domains: {domains}.
Query: {query}

Please provide a polite response explaining that the system works best with single-domain tasks,
and suggest a bulleted breakdown of 2-3 simpler sub-tasks the user can ask individually."""
    
    response = mgr.generate_supervisor(
        breakdown_prompt,
        system="You are a helpful routing assistant.",
        max_tokens=512,
        temperature=0.3,
        think_budget=THINK_BUDGET_UTIL
    )
    
    if session_store:
        session_store.write_task(query, "TIER3_STUB", domains, response, {"source": "stubbed_breakdown"})
        
    return response




