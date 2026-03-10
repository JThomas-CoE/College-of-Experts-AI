
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

# PHASE 2 — Two-Stage Tier Classifier
# ═══════════════════════════════════════════════════════════════════════════

DOMAIN_KEYWORDS: Dict[str, set] = {
    "code":    {"python", "function", "class", "algorithm", "implement", "code", "script",
                "debug", "refactor", "unit test", "async", "loop", "decorator", "module"},
    "sql":     {"sql", "query", "database", "table", "schema", "join", "select", "insert",
                "index", "postgresql", "mysql", "sqlite"},
    "math":    {"math", "calculus", "equation", "matrix", "probability", "integral",
                "derivative", "statistics", "proof", "theorem", "algebra"},
    "medical": {"medical", "clinical", "patient", "diagnosis", "treatment", "drug",
                "symptom", "disease", "health", "anatomy", "pharmacology"},
    "legal":   {"legal", "contract", "clause", "liability", "statute", "regulation",
                "court", "plaintiff", "defendant", "jurisdiction", "compliance",
                "law", "murder", "theft", "robbery", "burglary", "assault",
                "homicide", "felony", "misdemeanor", "crime", "criminal",
                "statute of limitations", "first degree", "second degree"},
    "web":     {"html", "css", "javascript", "react", "vue", "frontend", "backend", "api",
                "rest", "typescript", "component", "dom", "responsive", "timer", "button",
                "theme", "dark theme", "ui"},
}

_TRIVIAL_PATTERNS = re.compile(
    r"^(hi|hello|hey|help|what can you do|thanks|thank you|bye|quit|exit)\??\.?$",
    re.IGNORECASE,
)

_GENERAL_Q_STARTS = (
    "what is ",
    "what's ",
    "what are ",
    "what color ",
    "what does ",
    "what do ",
    "who is ",
    "who was ",
    "when is ",
    "when was ",
    "where is ",
    "where are ",
    "why is ",
    "why are ",
    "how fast ",
    "how far ",
    "how many ",
    "how much ",
    "how old ",
    "how big ",
    "how tall ",
    "define ",
    "explain ",
    "is the ",
    "are the ",
    "can a ",
    "can an ",
    "does the ",
    "do the ",
)


def classify_query(
    query: str,
    mgr: "ModelManager",
    confirm_enabled: bool = True,
) -> Tuple[str, List[str], str]:
    """
    Two-stage classifier.
    Returns (tier, domains, interpretation).
    Never returns empty domains — fallback is ("TIER2", ["supervisor"], "General query").
    """
    valid_domains = set(DOMAIN_MODELS.keys())
    ql = query.lower().strip()
    words = query.split()

    def _is_general_factual_query(text: str) -> bool:
        t = (text or "").strip().lower()
        if not t or len(t.split()) > 18:
            return False
        if not t.endswith("?") and not any(t.startswith(p) for p in _GENERAL_Q_STARTS):
            return False
        if not any(t.startswith(p) for p in _GENERAL_Q_STARTS):
            return False
        disqualifiers = (
            "write ",
            "build ",
            "create ",
            "implement ",
            "design ",
            "fix ",
            "debug ",
            "refactor ",
            "generate ",
            "story",
            "poem",
            "html",
            "sql",
            "function",
            "code",
            "api",
            "javascript",
            "css",
            "react",
            "typescript",
            "frontend",
            "backend",
            "database",
            "query",
            "contract",
            "clause",
            "diagnosis",
        )
        return not any(marker in t for marker in disqualifiers)

    def _has_explicit_build_or_edit_intent(text: str) -> bool:
        t = (text or "").strip().lower()
        intent_markers = (
            "write ",
            "build ",
            "create ",
            "implement ",
            "design ",
            "generate ",
            "make ",
            "fix ",
            "debug ",
            "refactor ",
            "update ",
            "modify ",
            "add ",
            "remove ",
            "convert ",
        )
        return any(t.startswith(marker) for marker in intent_markers)

    def _normalize_domains(raw_domains: List[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for item in raw_domains or []:
            dom = str(item).strip().lower()
            if dom in valid_domains and dom not in seen:
                normalized.append(dom)
                seen.add(dom)
        return normalized

    def _enforce_routing_contract(tier_in: str, domains_in: List[str]) -> Tuple[str, List[str]]:
        tier_norm = (tier_in or "TIER2").upper()
        doms_norm = _normalize_domains(domains_in)
        if not doms_norm:
            doms_norm = ["supervisor"]

        # Hard guard: TIER1 is only for truly trivial supervisor-only interactions.
        # Any domain-directed or non-trivial request must route to TIER2/TIER3.
        if tier_norm == "TIER1":
            if doms_norm != ["supervisor"] or not (_TRIVIAL_PATTERNS.match(ql) or _is_general_factual_query(ql)):
                if len(doms_norm) >= 2:
                    return "TIER3", doms_norm
                return "TIER2", doms_norm

        if tier_norm != "TIER1" and len(doms_norm) >= 2:
            return "TIER3", doms_norm
        if tier_norm == "TIER3" and len(doms_norm) < 2:
            return "TIER2", doms_norm
        if tier_norm not in ("TIER1", "TIER2", "TIER3"):
            return "TIER2", doms_norm
        return tier_norm, doms_norm

    def _is_veterinary_query(text: str) -> bool:
        animal_terms = {
            "cat", "cats", "dog", "dogs", "kitten", "kittens", "puppy", "puppies",
            "pet", "pets", "feline", "canine", "rabbit", "hamster", "bird", "parrot",
            "horse", "cow", "goat", "sheep", "veterinary", "vet",
        }
        health_terms = {
            "temperature", "fever", "symptom", "symptoms", "normal", "body temperature",
            "dose", "dosage", "medicine", "medication", "illness", "disease", "vomiting",
            "diarrhea", "injury", "pain", "heart rate", "breathing", "health",
        }
        has_animal = any(_keyword_match(term, text) for term in animal_terms)
        has_health = any(_keyword_match(term, text) for term in health_terms)
        return has_animal and has_health

    def _is_single_artifact_web_task(text: str) -> bool:
        t = (text or "").lower()
        has_web = any(k in t for k in ["html", "css", "javascript", "web app", "webapp", "timer", "button"])
        asks_single = any(k in t for k in ["self contained", "self-contained", "complete", "single html", "one file"])
        return has_web and asks_single

    def _is_creative_writing_query(text: str) -> bool:
        t = (text or "").lower().strip()
        creative_markers = [
            "write a short story",
            "write me a story",
            "tell me a story",
            "tell me a short story",
            "short story",
            "fiction",
            "poem",
            "haiku",
            "sonnet",
            "creative writing",
            "write a narrative",
            "write a poem",
            "write a haiku",
            "write a limerick",
            "a short story",
        ]
        return any(marker in t for marker in creative_markers)

    def _keyword_match(keyword: str, text: str) -> bool:
        """Match keywords by token boundary where possible to avoid substring collisions.
        Example: "functional" should not match keyword "function".
        """
        kw = keyword.strip().lower()
        if not kw:
            return False
        pattern = r'(?<![a-zA-Z0-9])' + re.escape(kw) + r'(?![a-zA-Z0-9])'
        return bool(re.search(pattern, text))

    # ── Stage 1: Heuristic scan ──────────────────────────────────────────
    # True greetings / meta-commands → TIER1 immediately, no supervisor call
    if _TRIVIAL_PATTERNS.match(ql):
        return "TIER1", ["supervisor"], query

    # Short general factual/science/reference questions should route directly
    # to supervisor TIER1 instead of entering stage-2 and drifting into absurd
    # multi-domain classifications.
    if _is_general_factual_query(ql) and not _has_explicit_build_or_edit_intent(ql):
        return "TIER1", ["supervisor"], f"A general factual question: {query}"

    # Veterinary health questions are safer on the supervisor path than on
    # the human-clinical specialist, and they should never escalate into a
    # spurious multi-domain route.
    if _is_veterinary_query(ql):
        return "TIER2", ["supervisor"], f"A veterinary health question: {query}"

    # Open-ended creative writing is handled directly by the supervisor in one
    # lean TIER1 pass — no specialist machinery needed, no grading, no retry.
    if _is_creative_writing_query(ql):
        return "TIER1", ["supervisor"], f"A creative writing request: {query}"

    scores: Dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in keywords if _keyword_match(kw, ql))
        if count:
            scores[domain] = count

    sorted_domains = sorted(scores, key=lambda d: scores[d], reverse=True)

    is_trivial = len(words) < 4  # short queries: skip high-confidence shortcut, go Stage 2
    is_multi = (
        len(sorted_domains) >= 2
        and len(words) > 40
    )

    # Determine confidence
    if len(sorted_domains) >= 2:
        top, second = scores[sorted_domains[0]], scores[sorted_domains[1]]
        high_confidence = top > 2 * second and not is_multi
    elif len(sorted_domains) == 1:
        high_confidence = not is_trivial and not is_multi
    else:
        high_confidence = False

    # TIER1 or TIER3 candidates → always go Stage 2
    if is_trivial:
        pass  # Stage 2 will classify short-but-meaningful queries (e.g. "write quicksort")
    elif is_multi:
        pass  # Stage 2 below
    elif high_confidence and sorted_domains:
        # TIER2 high confidence — skip Stage 2
        interpretation = f"A {sorted_domains[0]} task: {query}"
        return "TIER2", [sorted_domains[0]], interpretation

    # ── Stage 2: Supervisor classification ──────────────────────────────
    prompt = (
        "Classify this query for routing. Output JSON with keys:\n"
        "  tier: \"TIER1\" | \"TIER2\" | \"TIER3\"\n"
        "  domains: [list of domain keys from: code, sql, math, medical, legal, web, supervisor]\n"
        "  interpretation: one-sentence summary of what is being asked\n"
        "  confidence: \"high\" | \"low\"\n"
        "  recheck_needed: true | false\n\n"
        "Rules:\n"
        "- Simple factual / everyday knowledge questions with no build/edit/artifact intent are TIER1 and domains [\"supervisor\"].\n"
        "- Only use domain 'web' for explicit web/app/UI/frontend/backend/API/HTML/CSS/JavaScript work.\n"
        "- Do not infer 'web' from generic words like color, appearance, look, style, or theme unless the user is clearly asking for a web artifact.\n\n"
        f"Query: {query}"
    )
    system = "You are a precise routing classifier. Output only valid JSON."

    def _run_stage2(qtext: str) -> Tuple[str, List[str], str]:
        try:
            raw = mgr.generate_supervisor(qtext, system=system, max_tokens=120, temperature=0.01, think_budget=THINK_BUDGET_UTIL)
            # Strip markdown code fences if present
            raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("```").strip()
            # Find JSON object
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                raise ValueError("No JSON found")
            parsed = json.loads(m.group(0))
            tier = str(parsed.get("tier", "TIER2")).upper()
            doms = [str(d).lower() for d in parsed.get("domains", [])]
            interp = str(parsed.get("interpretation", query))
            recheck = parsed.get("recheck_needed", False)
            tier, doms = _enforce_routing_contract(tier, doms)
            return tier, doms, interp, recheck
        except Exception as e:
            cprint(f"[Classifier] Stage 2 parse error: {e}", "yellow")
            return "TIER2", ["supervisor"], query, False

    tier, doms, interp, recheck = _run_stage2(prompt)

    # One recheck if supervisor flagged it
    if recheck:
        recheck_prompt = (
            f"Re-classify with more precision. Previous domains: {doms}.\n\n{prompt}"
        )
        tier, doms, interp, _ = _run_stage2(recheck_prompt)

    tier, doms = _enforce_routing_contract(tier, doms)

    # Guard: if stage-2 overclassifies a short factual question into a large
    # multi-domain set, collapse it back to direct supervisor handling.
    if _is_general_factual_query(ql) and (len(doms) >= 2 or set(doms) == valid_domains):
        return "TIER1", ["supervisor"], f"A general factual question: {query}"

    # Guard: simple informational questions with weak/no heuristic domain evidence
    # should not drift into a specialist domain due to stage-2 over-inference.
    if _is_general_factual_query(ql) and not _has_explicit_build_or_edit_intent(ql):
        max_score = max(scores.values()) if scores else 0
        if max_score <= 1 and doms != ["supervisor"]:
            return "TIER1", ["supervisor"], f"A general factual question: {query}"

    # Guard: single-artifact HTML web tasks should not split into code+web TIER3.
    if tier == "TIER3" and set(doms) == {"code", "web"} and _is_single_artifact_web_task(query):
        return "TIER2", ["web"], interp

    # BioMistral is tuned for human clinical text. Veterinary questions are
    # safer routed through the supervisor-only path unless a dedicated animal
    # health specialist exists.
    if "medical" in doms and _is_veterinary_query(ql):
        return "TIER2", ["supervisor"], f"A veterinary health question: {query}"

    # Hard specialist guard: if heuristic scoring strongly signals a specialist
    # domain (score >= 2 keyword hits) but Stage 2 classified it as supervisor-only,
    # override back to the specialist.  The supervisor is a small reasoning model —
    # it must never answer production code or web tasks.
    if doms == ["supervisor"] and sorted_domains:
        top_domain = sorted_domains[0]
        if scores.get(top_domain, 0) >= 2 and top_domain in {"code", "web"}:
            return "TIER2", [top_domain], interp

    return tier, doms, interp


# ═══════════════════════════════════════════════════════════════════════════




