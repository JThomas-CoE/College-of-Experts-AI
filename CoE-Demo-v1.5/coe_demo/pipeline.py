
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
from .router import *
from .kb import *

# PHASE 3 — Qualifier Infrastructure & Pipelines
# ═══════════════════════════════════════════════════════════════════════════

QUALIFIER_KEYWORDS: Dict[str, Dict[str, str]] = {
    "language": {
        "python": "Python", "c++": "C++", "cpp": "C++", "c#": "C#", "csharp": "C#",
        "java": "Java", "javascript": "JavaScript", "typescript": "TypeScript",
        "go": "Go", "golang": "Go", "rust": "Rust", "swift": "Swift",
        "kotlin": "Kotlin", "ruby": "Ruby", "php": "PHP",
    },
    "web_framework": {
        "react": "React", "vue": "Vue", "angular": "Angular", "svelte": "Svelte",
        "next.js": "Next.js", "nuxt": "Nuxt", "htmx": "HTMX",
    },
    "sql_dialect": {
        "postgresql": "PostgreSQL", "postgres": "PostgreSQL", "mysql": "MySQL",
        "sqlite": "SQLite", "mssql": "SQL Server", "bigquery": "BigQuery",
        "snowflake": "Snowflake",
    },
    "subspecialty": {
        "cardiology": "cardiology", "oncology": "oncology", "pediatrics": "pediatrics",
        "pharmacology": "pharmacology", "neurology": "neurology",
        "gdpr": "GDPR/EU compliance", "hipaa": "HIPAA/US healthcare law",
        "uk law": "UK law", "eu law": "EU law", "ip law": "IP law",
        "contract law": "contract law",
    },
    "style": {
        "with unit tests": "with unit tests", "with tests": "with unit tests",
        "with docstrings": "with docstrings", "production-grade": "production-grade",
        "production grade": "production-grade", "functional": "functional style",
        "beginner-friendly": "beginner-friendly", "beginner": "beginner-friendly",
        "concise": "concise", "step-by-step": "step-by-step",
        "with comments": "with inline comments",
    },
    "level": {
        "advanced": "advanced", "intermediate": "intermediate",
        "expert": "expert-level", "academic": "academic/formal",
    },
}


@dataclass
class QueryQualifiers:
    language:     Optional[str] = None
    framework:    Optional[str] = None
    dialect:      Optional[str] = None
    subspecialty: Optional[str] = None
    style:        Optional[str] = None
    level:        Optional[str] = None

    def summary(self) -> str:
        parts = [f"{k}={v}" for k, v in [
            ("language", self.language), ("framework", self.framework),
            ("dialect", self.dialect), ("subspecialty", self.subspecialty),
            ("style", self.style), ("level", self.level),
        ] if v]
        return ", ".join(parts) if parts else "none"

    def has_any(self) -> bool:
        return any([self.language, self.framework, self.dialect,
                    self.subspecialty, self.style, self.level])


def extract_qualifiers(query: str) -> QueryQualifiers:
    """Rule-based qualifier extraction. Zero model cost. Case-insensitive.
    Uses alphanumeric boundaries (?<![a-zA-Z0-9]) so C++, C#, Next.js etc. match correctly."""
    ql = query.lower()
    q = QueryQualifiers()

    def _word_match(kw: str, text: str) -> bool:
        """Match keyword not adjacent to alphanumeric characters."""
        return bool(re.search(
            r'(?<![a-zA-Z0-9])' + re.escape(kw) + r'(?![a-zA-Z0-9])',
            text
        ))

    for kw, val in QUALIFIER_KEYWORDS["language"].items():
        if _word_match(kw, ql):
            q.language = val; break
    for kw, val in QUALIFIER_KEYWORDS["web_framework"].items():
        if kw in ql: q.framework = val; break
    for kw, val in QUALIFIER_KEYWORDS["sql_dialect"].items():
        if kw in ql: q.dialect = val; break
    for kw, val in QUALIFIER_KEYWORDS["subspecialty"].items():
        if kw in ql: q.subspecialty = val; break
    for kw, val in QUALIFIER_KEYWORDS["style"].items():
        if kw in ql: q.style = val; break
    for kw, val in QUALIFIER_KEYWORDS["level"].items():
        if _word_match(kw, ql):
            q.level = val; break
    return q


SPECIALIST_PERSONA_TEMPLATES: Dict[str, str] = {
    "code":       "You are an expert {language} engineer specializing in clean, efficient, well-documented code.{style_clause}",
    "sql":        "You are an expert {dialect} database architect specializing in query optimization and schema design.",
    "math":       "You are an expert mathematician providing rigorous, step-by-step solutions.{level_clause}",
    "medical":    "You are an expert clinician{subspecialty_clause} providing evidence-based clinical information.",
    "legal":      "You are an expert legal analyst specializing in {subspecialty_or_default}.{level_clause}",
    "web":        "You are an expert {framework} developer specializing in modern web applications and REST APIs.{style_clause}",
    "supervisor": "You are a knowledgeable generalist assistant.",
}


def patch_persona(domain: str, q: QueryQualifiers) -> str:
    """Instantiate specialist persona template with detected qualifiers."""
    t = SPECIALIST_PERSONA_TEMPLATES.get(domain, SPECIALIST_PERSONA_TEMPLATES["supervisor"])
    return t.format(
        language=q.language or "software",
        dialect=q.dialect or "relational",
        framework=(q.framework or "full-stack web") + " and REST API",
        subspecialty_clause=f" specializing in {q.subspecialty}" if q.subspecialty else "",
        subspecialty_or_default=q.subspecialty or "contract law and regulatory compliance",
        style_clause=f" Focus on {q.style}." if q.style else "",
        level_clause=f" Tailor for {q.level} audience." if q.level else "",
    )


def build_supervisor_answer_system(base_system: str, domain: str) -> str:
    """Add a domain-grounded honesty preface for supervisor answer generation."""
    domain_label = (domain or "general topics").replace("_", " ")
    preface = (
        f"You are an expert in {domain_label} and able to give grounded, well-thought-out "
        f"information about {domain_label}. However, you are also honest and willing to tell "
        "the user if you are not sure of an answer or if there are substantial and serious "
        "conflicting schools of thought about a particular subject."
    )
    base = (base_system or "").strip()
    if not base:
        return preface
    return f"{preface}\n\n{base}"


def build_supervisor_tier2_specialist_system(base_system: str, domain: str) -> str:
    """Stricter system prompt for supervisor acting in TIER2 specialist capacity."""
    domain_label = (domain or "general topics").replace("_", " ")
    shared_rules = (
        f"You are acting as the {domain_label} specialist for this request. "
        "Be precise, grounded, and conservative. Prefer a shorter accurate answer over a longer speculative one. "
        "Do not invent citations, statutes, case names, technical terms, or factual details. "
        "If a detail is uncertain, say that it varies by jurisdiction, source, version, or context rather than guessing. "
        "Do not pad the answer with broad comparisons unless they are necessary to answer the user's actual question. "
        "Do not include meta-commentary about your process."
    )

    domain_rules = {
        "legal": (
            "For legal topics, do not present legal information as universal when it is jurisdiction-dependent. "
            "Explicitly say when a concept varies by jurisdiction. "
            "If the user did not specify a jurisdiction, give only a clearly labeled general overview and note the limitation. "
            "Do not cite statutes, section numbers, regulations, cases, or named legal tests unless the user asked for a specific jurisdiction and you are confident they are real and relevant. "
            "Do not provide a multi-country or multi-state survey unless the user explicitly asks for comparison. "
            "Avoid edge-case exceptions, sentencing details, historical background, and illustrative cases unless they are necessary to answer the user's actual question. "
            "Prefer a short high-level definition with a brief note that details vary by jurisdiction."
        ),
        "medical": (
            "For medical topics, distinguish clearly between general educational information and individualized medical advice. "
            "Do not fabricate study findings, dosages, contraindications, or diagnostic certainty."
        ),
        "code": (
            "For code tasks, satisfy the requested language and output format exactly. "
            "Do not switch languages or add explanatory prose when code-only output is requested."
        ),
    }

    base = build_supervisor_answer_system(base_system, domain)
    extra = domain_rules.get(domain, "")
    return f"{base}\n\n{shared_rules}" + (f"\n\n{extra}" if extra else "")


# ── TIER1 Pipeline ────────────────────────────────────────────────────────

_CREATIVE_TIER1_MARKERS = (
    "story", "poem", "haiku", "sonnet", "limerick", "narrative", "fiction",
    "creative writing",
)

def _is_creative_tier1(query: str) -> bool:
    q = query.lower()
    return any(m in q for m in _CREATIVE_TIER1_MARKERS)

def _extract_length_qualifier(query: str) -> Optional[str]:
    """Return a length directive if the user explicitly asked for one."""
    q = query.lower()
    if "very short" in q or "tiny" in q or "one sentence" in q:
        return "very short (1-2 paragraphs maximum)"
    if "short" in q or "brief" in q or "quick" in q:
        return "short (3-4 paragraphs maximum)"
    if "long" in q or "detailed" in q or "extended" in q:
        return None  # let the model decide
    return None  # default: no constraint


def _stream_token(text: str) -> None:
    """Default streaming callback — writes directly to stdout without buffering."""
    import sys
    sys.stdout.write(text)
    sys.stdout.flush()


# ── Follow-up context helpers ─────────────────────────────────────────────────

_NEW_TASK_MARKERS = (
    "make me a ", "make a ", "create a ", "build a ", "write a ",
    "generate a ", "design a ", "implement a ", "make a new ", "create a new ",
    "i need a ", "can you make a ", "can you build a ", "please create a ",
    "please make a ", "please write a ", "please build a ", "produce a ",
)


def _is_clearly_new_task(query: str) -> bool:
    """Fast Python heuristic (~0ms) — returns True when the query contains
    a self-contained task description with an explicit new-subject noun.
    Catches obvious cases without touching the supervisor.
    """
    q = query.lower().strip()
    return any(q.startswith(m) or f" {m}" in q for m in _NEW_TASK_MARKERS)


def _build_followup_specialist_system(
    base_system: str,
    prior_query: str,
    prior_artifact: str,
    max_artifact_chars: int = 18000,
) -> str:
    """Appends the prior artifact to the specialist system prompt for follow-up queries.

    Placing prior content in the SYSTEM prompt (not the user prompt) prevents the
    model from echoing or continuing it as new output. It is treated as reference
    material, not as conversational text to complete.
    """
    artifact = prior_artifact[:max_artifact_chars]
    if len(prior_artifact) > max_artifact_chars:
        artifact += "\n...[truncated — full artifact in session store]..."
    return (
        base_system
        + "\n\n# PRIOR TASK CONTEXT"
        + "\n# The user is requesting a modification or fix to the following prior output."
        + "\n# CRITICAL RULE: You must output the ENTIRE modified artifact in the exact same format as the original."
        + "\n# Do NOT provide partial snippets, diffs, conversational advice, or explanations."
        + "\n# Apply the requested changes and output the complete, standalone result."
        + f"\n# Prior user query: \"{prior_query.strip()}\""
        + f"\n# Prior output:\n{artifact}"
    )



def run_tier1(query: str, mgr: ModelManager) -> str:
    """TIER1: supervisor answers directly with streaming output.

    Creative writing requests get a tailored system prompt that bans
    meta-commentary, footnotes and postscript analysis, and honours any
    explicit length qualifier in the user's request.
    """
    if _is_creative_tier1(query):
        length_qualifier = _extract_length_qualifier(query)
        length_clause = (
            f" Keep the response {length_qualifier}."
            if length_qualifier else ""
        )
        system = (
            "You are a creative writer. Respond with the requested content only. "
            "Do NOT add any analysis, commentary, footnotes, postscripts, thematic summaries, "
            "or meta-text after the creative piece ends. "
            "End the response immediately when the story, poem, or creative piece is complete. "
            "No '---' separators. No '* This story reflects...' type additions."
            f"{length_clause}"
        )
        think_budget = THINK_BUDGET_T1  # minimal thinking for creative tasks
        max_tokens = OUTPUT_BUDGET_T1
    else:
        system = build_supervisor_answer_system(
            (
                "You are a helpful, knowledgeable assistant. "
                "Answer clearly and concisely."
            ),
            "general topics",
        )
        think_budget = THINK_BUDGET_T1
        max_tokens = OUTPUT_BUDGET_T1

    cprint("\nResponse", "bold")
    result = generate_supervisor_answer_resilient(
        mgr=mgr,
        prompt=query,
        system=system,
        max_tokens=max_tokens,
        temperature=0.7,
        think_budget=think_budget,
        retry_label="tier1",
        on_token=_stream_token,
    )
    print()  # newline after streaming
    return result



# ── TIER2 Pipeline ────────────────────────────────────────────────────────

def run_tier2(
    query: str,
    domain: str,
    mgr: ModelManager,
    session_store: "SessionStore",
    template_store: "TemplateStore",
    skill_store: "SkillStore" = None,
    enhance: bool = True,
    kb_enabled: bool = True,
    self_grade: bool = True,
    render_mode: str = "intermediate+final",
    resume_after_switch: bool = False,
    router_followup: bool = False,
    loop_guard: bool = True,
) -> str:
    """
        TIER2 benchmark-style pipeline.
      0  Extract qualifiers + patch persona
      1  KB pre-check (supervisor decides)
            2  Ensure specialist + supervisor prompt formulation
            3  Specialist draft generation
            4  Quality-gated single retry (if needed)
            5  Keep better specialist candidate
            6  Supervisor synthesis and keep better final answer
    """
    show_intermediate = normalize_render_mode(render_mode) != "final only"
    task_key = session_store.peek_task_key() if session_store else f"adhoc:{int(time.time())}"
    phase_artifact_ids: Dict[str, str] = {}

    # Step 0 — Qualifier extraction
    qualifiers = extract_qualifiers(query)
    specialist_system = patch_persona(domain, qualifiers)
    supervisor_answer_system = build_supervisor_answer_system(specialist_system, domain)
    supervisor_tier2_system = build_supervisor_tier2_specialist_system(specialist_system, domain)
    if qualifiers.has_any():
        cprint(f"[CoE] Qualifiers: {qualifiers.summary()}", "dim")

    def _fallback_specialist_task_prompt(
        kb_context: Optional[str] = None,
        template_match=None,
    ) -> str:
        """Build the user-facing task prompt — query + KB + template only.
        Session context is never injected: each generation call is stateless.
        """
        template_injection = ""
        if template_match:
            imperative = TEMPLATE_IMPERATIVE[template_match.strength]
            template_injection = f"\n\n{imperative}\n{template_match.scaffold_text}"
        kb_block = f"\n\nKB reference:\n{kb_context}" if kb_context else ""
        return f"{query}{kb_block}{template_injection}".strip()

    def _looks_like_prompt_scaffold(text: str) -> bool:
        lowered = (text or "").strip().lower()
        if not lowered:
            return True
        suspicious_markers = [
            "output only the prompt text",
            "construct an optimal prompt",
            "context for code specialist",
            "insert missing context here",
            "task:",
            "instructions:",
        ]
        if lowered.startswith("prompt"):
            return True
        return sum(1 for marker in suspicious_markers if marker in lowered) >= 2

    def _run_zero_shot_supervisor_specialist(
        reason: str,
        kb_context: Optional[str] = None,
        template_match=None,
    ) -> str:
        cprint(f"[LoadPolicy] {reason} — defaulting to supervisor-only zero-shot specialist harness.", "yellow")
        template_injection = ""
        if template_match:
            imperative = TEMPLATE_IMPERATIVE[template_match.strength]
            template_injection = f"\n\n{imperative}\n{template_match.scaffold_text}"
        kb_block = f"\n\nReference:\n{kb_context}" if kb_context else ""
        task_prompt = (
            f"Answer the user's request directly as the {domain} specialist."
            f" Do not write meta-instructions, prompt text, or analysis headers."
            f"\n\nUser request: {query}"
            f"{kb_block}{template_injection}"
        )
        cprint("\nResponse (Supervisor — Synthetic Specialist)", "bold")
        result = generate_supervisor_answer_resilient(
            mgr=mgr,
            prompt=task_prompt,
            system=supervisor_tier2_system,
            max_tokens=SUPERVISOR_OUTPUT_BUDGET_T2,
            temperature=0.2,
            think_budget=THINK_BUDGET_T2,
            retry_label=f"tier2_synthetic_{domain}",
            on_token=_stream_token,
        )
        print()
        if session_store:
            artifacts = {"generation_artifact_ids": phase_artifact_ids}
            tier = "TIER2_SYNTHETIC"
            session_store.write_task(query, tier, [domain], result, artifacts)
        return result

    # ── Follow-up vs. New-task detection ─────────────────────────────────────
    # Each Ollama/ONNX call is stateless (no KV carry-over). Prior task text lives
    # in Python RAM only. We inject it into the specialist system prompt ONLY when
    # the supervisor confirms this query is a follow-up/refinement of the prior task.
    #
    # Layer 1 — fast Python heuristic (~0ms): catches obvious new tasks
    # Layer 2 — supervisor YES/NO (~1-2s, UTIL budget, no thinking): resolves ambiguous cases
    # If FOLLOWUP: prior artifact → specialist system prompt only (not user prompt)
    # If NEW_TASK: clean stateless call
    # Use get_recent_specialist_tasks() — not get_recent_tasks() — so TIER1
    # supervisor responses are excluded at the API boundary, not by ad-hoc guards.
    _prior_tasks = session_store.get_recent_specialist_tasks(1) if session_store else []
    _prior_task = _prior_tasks[0] if _prior_tasks else None
    if _prior_task:
        _prior_query = str(_prior_task.get("query", ""))
        _prior_artifact = session_store._accepted_text_for_task(_prior_task) if _prior_query else ""
        # Defense-in-depth: if anything slips through the specialist filter, block it here.
        _prior_tier_str = str(_prior_task.get("tier", "TIER2")).upper()
        _prior_domains_list = [str(d).lower() for d in (_prior_task.get("domains") or [])]
        if _prior_tier_str.startswith("TIER1") or _prior_domains_list == ["supervisor"]:
            _prior_artifact = ""
            cprint("[Context] Prior task was TIER1/supervisor — skipping artifact injection.", "dim")
        # Domain mismatch: a prior code artifact is irrelevant to the web specialist
        # and vice versa.  Artifact injection is only logically valid when continuing
        # work in the same specialist domain.  If the user has switched domains the
        # prior output is out-of-scope and must not be injected unless they explicitly
        # reference it (which they can do in their query text).
        _prior_domain = _prior_domains_list[0] if _prior_domains_list else ""
        if _prior_artifact and _prior_domain and _prior_domain != domain.lower():
            _prior_artifact = ""
            cprint(f"[Context] Domain changed ({_prior_domain} \u2192 {domain}) — prior artifact not applicable.", "dim")
        if router_followup and _prior_artifact:
            cprint("[Context] Follow-up confirmed (RoutingGuard).", "dim")
            cprint(
                f"[Context] Injecting prior artifact "
                f"({min(len(_prior_artifact), 8000)} chars) into specialist context.",
                "dim",
            )
            specialist_system = _build_followup_specialist_system(
                specialist_system, _prior_query, _prior_artifact
            )
        elif _is_clearly_new_task(query):
            cprint("[Context] New standalone task (heuristic) — clean context.", "dim")
        elif _prior_artifact:
            # Ambiguous — supervisor decides; runs in parallel with specialist load
            cprint("[Context] Checking follow-up status...", "dim")
            _classification = classify_followup(query, _prior_query, mgr)
            if _classification == "FOLLOWUP":
                cprint(
                    f"[Context] Follow-up confirmed — injecting prior artifact "
                    f"({min(len(_prior_artifact), 18000)} chars) into specialist context.",
                    "dim",
                )
                specialist_system = _build_followup_specialist_system(
                    specialist_system, _prior_query, _prior_artifact
                )
                # NOTE: supervisor_tier2_system and supervisor_answer_system are NOT
                # updated — supervisor paths run clean (limited context window).
            elif _classification == "NEW_TASK":
                cprint("[Context] New task (supervisor) — clean context.", "dim")
            else:
                cprint(f"\n[CoE] Unsure if this request continues the previous task or is entirely new.", "yellow")
                cprint(f"      Prior task: \"{_prior_query[:100]}...\"", "yellow")
                try:
                    ans = input("  Does this modify or reference the prior output? [y/N] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    ans = "n"
                if ans.startswith("y"):
                    cprint("[Context] Proceeding as follow-up task.", "dim")
                    specialist_system = _build_followup_specialist_system(
                        specialist_system, _prior_query, _prior_artifact
                    )
                else:
                    cprint("[Context] Proceeding as new task with clean context.", "dim")

    # Template pre-check
    match = None
    if template_store and mgr._embedder:
        match = template_store.find_match(query, mgr._embedder)
        if match:
            cprint(f"[TEMPLATE] {match.strength.upper()} match: {match.title} ({match.similarity:.2f})")

    # Skill pre-check — inject advisory guidance into specialist_system
    _skill_match = None
    if skill_store and mgr._embedder and domain not in ("supervisor",):
        _skill_match = skill_store.find_match(query, domain, mgr._embedder)
        if _skill_match:
            cprint(f"[SKILL] {_skill_match.title} ({_skill_match.similarity:.2f})", "dim")
            specialist_system = (
                specialist_system
                + f"\n\nAdvisory guidance for this task:\n{_skill_match.guidance}"
            )

    if domain == "supervisor":
        cprint("[LoadPolicy] Supervisor-routed TIER2 request — answering directly with supervisor.", "dim")
        template_injection = ""
        if match:
            imperative = TEMPLATE_IMPERATIVE[match.strength]
            template_injection = f"\n\n{imperative}\n{match.scaffold_text}"
        task_prompt = (
            "Answer the user's request directly. "
            "Do not write meta-instructions, prompt text, or analysis headers."
            f"\n\nUser request: {query}"
            f"{template_injection}"
        )
        # Direct supervisor route: single substantive answer pass.
        # Use SUPERVISOR_OUTPUT_BUDGET_T2 — ONNX/DML pre-allocates KV sized to max_length.
        cprint("\nResponse (Supervisor)", "bold")
        result = generate_supervisor_answer_resilient(
            mgr=mgr,
            prompt=task_prompt,
            system=supervisor_tier2_system,
            max_tokens=SUPERVISOR_OUTPUT_BUDGET_T2,
            temperature=0.2,
            think_budget=THINK_BUDGET_T2,
            retry_label=f"tier2_supervisor_{domain}",
            on_token=_stream_token,
        )
        print()
        if session_store:
            artifacts = {"generation_artifact_ids": phase_artifact_ids}
            session_store.write_task(query, "TIER2", [domain], result, artifacts)
        return result

    # Step 1 — Benchmark-style direct specialist prompt path.
    # Avoid pre-specialist supervisor calls; they have proven unstable across
    # domain transitions in this long-lived CLI process.
    kb_context = None
    if resume_after_switch:
        cprint("[LoadPolicy] Benchmark-style resumed switch: skipping supervisor pre-processing before specialist load.", "dim")

    synthetic_reason = ""

    # Step 2 — Background specialist load + supervisor prompt formulation
    # 'supervisor' domain always runs through the supervisor directly (no separate specialist)
    if domain in DISABLED_SPECIALIST_DOMAINS:
        cprint(f"[LoadPolicy] {domain} specialist disabled on this build — using supervisor-driven synthetic specialist.", "yellow")
    _raw = DOMAIN_MODELS.get(domain) if domain != "supervisor" and domain not in DISABLED_SPECIALIST_DOMAINS else None
    specialist_model = mgr._resolve(_raw) if _raw else None
    is_synthetic = specialist_model is None
    defer_specialist_load = False

    if is_synthetic:
        synthetic_reason = f"no domain model available for '{domain}'"
        cprint(f"[SYNTHETIC SPECIALIST — {synthetic_reason}]", "yellow")
    else:
        current_specialist = mgr.specialist_path()
        if current_specialist and current_specialist != specialist_model:
            defer_specialist_load = True
            cprint(
                f"[LoadPolicy] Serializing specialist switch for {Path(specialist_model).name} "
                f"until supervisor formulation completes.",
                "dim",
            )
            debug_log(
                "specialist.load_deferred",
                current=Path(current_specialist).name,
                target=Path(specialist_model).name,
                reason="cross-domain-switch",
            )
        else:
            mgr.ensure_specialist_for_model(specialist_model)

    if enhance and not resume_after_switch and is_synthetic:
        if is_synthetic:
            return _run_zero_shot_supervisor_specialist(synthetic_reason or f"synthetic {domain} path", kb_context, match)
        template_injection = ""
        if match:
            imperative = TEMPLATE_IMPERATIVE[match.strength]
            template_injection = f"\n\n{imperative}\n{match.scaffold_text}"

        kb_block = f"\n\nKB reference:\n{kb_context}" if kb_context else ""
        qual_note = f" Specialist qualifiers detected: {qualifiers.summary()}." if qualifiers.has_any() else ""

        formulation_system = (
            f"You are a precise prompt engineer. Construct an optimal prompt for a {domain} "
            f"specialist to answer the following query.{qual_note} Incorporate any provided context. "
            f"Output only the prompt text."
        )
        formulation_input = (
            f"Original query: {query}"
            f"{kb_block}{template_injection}"
        )
        enriched_prompt = mgr.generate_supervisor(
            formulation_input, system=formulation_system, max_tokens=512, temperature=0.2,
            think_budget=THINK_BUDGET_UTIL,
        )
        if _looks_like_prompt_scaffold(enriched_prompt):
            cprint("[LoadPolicy] Discarding prompt-scaffold enhancement and using raw specialist task prompt.", "yellow")
            debug_log("tier2.prompt_scaffold_discarded", domain=domain)
            enriched_prompt = _fallback_specialist_task_prompt(kb_context, match)
    else:
        if is_synthetic:
            return _run_zero_shot_supervisor_specialist(synthetic_reason or f"synthetic {domain} path", kb_context, match)
        enriched_prompt = _fallback_specialist_task_prompt(kb_context, match)

    if resume_after_switch and not is_synthetic:
        enriched_prompt = _fallback_specialist_task_prompt(kb_context, match)

    if not is_synthetic and defer_specialist_load:
        mgr.ensure_specialist_for_model(specialist_model)

    if domain == "code":
        code_language = qualifiers.language or "requested"
        if wants_code_only(query):
            enriched_prompt += (
                f"\n\nOutput requirements: Return ONLY one {code_language} function definition. "
                "No prose, no explanation, no markdown fences, no comments, no docstring."
            )
        else:
            enriched_prompt += (
                f"\n\nOutput requirements: Start with a valid {code_language} function definition. "
                "Keep any extra explanation minimal."
            )

    if not is_synthetic:
        mgr.await_specialist()
        if mgr._load_error:
            cprint(f"[TIER2] Specialist load failed: {mgr._load_error} — using synthetic path", "yellow")
            is_synthetic = True
            return _run_zero_shot_supervisor_specialist(f"specialist load failed for '{domain}'", kb_context, match)

    # Step 3 — Specialist draft generation
    if is_synthetic:
        # Supervisor (ONNX/DML): use capped budget to prevent DML arena inflation
        if show_intermediate:
            cprint("\nResponse (Draft)", "bold")
        draft = mgr.generate_supervisor(enriched_prompt, system=supervisor_tier2_system,
                                        max_tokens=SUPERVISOR_OUTPUT_BUDGET_T2, temperature=0.3,
                                        think_budget=THINK_BUDGET_T2,
                                        on_token=_stream_token if show_intermediate else None)
        if show_intermediate:
            print()
    else:
        # Ollama specialist: clean stateless call — query only, no prior context
        if show_intermediate:
            cprint("\nResponse (Draft)", "bold")
        draft = mgr.generate_specialist(enriched_prompt, system=specialist_system,
                                        max_tokens=OUTPUT_BUDGET_T2,
                                        on_token=_stream_token if show_intermediate else None,
                                        loop_guard=loop_guard)
        if show_intermediate:
            print()
    if domain == "code":
        draft = normalize_python_function_output(query, draft)
    # render_response skipped — draft was streamed live above

    # Step 4 — Quality evaluation
    grade_system = "You are a strict technical evaluator. Output only a verdict line."
    draft_verdict_word, draft_verdict_line = grade_candidate(
        query, draft, domain, mgr, grade_system,
        self_grade=self_grade,
        use_specialist_self_grade=((not is_synthetic) and domain in SPECIALIST_SELF_GRADE_DOMAINS),
        skill_guidance=_skill_match.guidance if _skill_match else None,
        template_scaffold=match.scaffold_text if (match and match.strength == "strict") else None,
    )
    cprint(f"[Grade] {draft_verdict_line}", "dim")

    draft_artifact_id = ""
    if session_store:
        draft_artifact_id = session_store.add_generation_artifact(
            task_key, "tier2_draft", draft, domain=domain, verdict=draft_verdict_word,
        )
        phase_artifact_ids["draft"] = draft_artifact_id

    best_specialist = draft
    best_specialist_word = draft_verdict_word
    best_specialist_line = draft_verdict_line
    best_specialist_artifact_id = draft_artifact_id

    # Step 5 — Single retry when specialist output fails quality gate
    if should_retry_tier2(draft_verdict_word, draft):
        fail_cat = draft_verdict_word if draft_verdict_word in FAIL_CATEGORIES else "FAIL_INCOMPLETE"
        # Strip the category prefix from the verdict line to get the raw reason clause
        fail_reason = draft_verdict_line[len(fail_cat):].lstrip(":").strip() if ":" in draft_verdict_line else "short output"
        cprint(f"[TIER2] Retry — {fail_cat}: {fail_reason}", "yellow")

        # KB retrieval: use the grader's specific fail_reason as the search key,
        # NOT the original query — the fail phrase targets the exact missing concept.
        retry_kb = None
        if kb_enabled:
            retry_kb = kb_retrieve_retry(fail_reason, domain)

        # Build retry prompt from pure Python template — NO Supervisor LLM call,
        # NO failed draft injected back into context.
        # The specialist receives: original_query + requirement hint + optional KB.
        retry_prompt = build_retry_prompt_from_template(
            original_query=query,
            fail_reason=fail_reason,
            kb_snippet=retry_kb,
        )

        # Code domain: append output-format requirement to retry prompt too
        if domain == "code":
            code_language = qualifiers.language or "requested"
            if wants_code_only(query):
                retry_prompt += (
                    f"\n\nOutput requirements: Return ONLY one {code_language} function definition. "
                    "No prose, no explanation, no markdown fences, no comments, no docstring."
                )
            else:
                retry_prompt += (
                    f"\n\nOutput requirements: Start with a valid {code_language} function definition. "
                    "Keep any extra explanation minimal."
                )
        elif domain == "web":
            retry_prompt += (
                f"\n\nOutput requirements: Return ONLY the complete, fully functional HTML file. "
                "Include all CSS and JS inline. Do NOT provide partial snippets, diffs, conversational advice, or explanations."
            )

        if is_synthetic:
            if show_intermediate:
                cprint("\nResponse (Retry)", "bold")
            retry_draft = mgr.generate_supervisor(retry_prompt, system=supervisor_tier2_system,
                                                  max_tokens=SUPERVISOR_OUTPUT_BUDGET_T2, temperature=0.4,
                                                  think_budget=THINK_BUDGET_T2,
                                                  on_token=_stream_token if show_intermediate else None)
            if show_intermediate:
                print()
        else:
            if not mgr.specialist_path():
                mgr.ensure_specialist_for_model(specialist_model)
                mgr.await_specialist()
            # Retry at T=0.9 — clean stateless call, no prior context
            if show_intermediate:
                cprint("\nResponse (Retry)", "bold")
            retry_draft = mgr.generate_specialist(retry_prompt, system=specialist_system,
                                                  max_tokens=OUTPUT_BUDGET_T2,
                                                  on_token=_stream_token if show_intermediate else None,
                                                  loop_guard=loop_guard)
            if show_intermediate:
                print()
        if domain == "code":
            retry_draft = normalize_python_function_output(query, retry_draft)
        # render_response skipped — retry was streamed live above

        retry_verdict_word, retry_verdict_line = grade_candidate(
            query, retry_draft, domain, mgr, grade_system,
            self_grade=self_grade,
            use_specialist_self_grade=((not is_synthetic) and domain in SPECIALIST_SELF_GRADE_DOMAINS),
            skill_guidance=_skill_match.guidance if _skill_match else None,
            template_scaffold=match.scaffold_text if (match and match.strength == "strict") else None,
        )
        cprint(f"[Grade retry] {retry_verdict_line}", "dim")

        retry_artifact_id = ""
        if session_store:
            retry_artifact_id = session_store.add_generation_artifact(
                task_key, "tier2_retry", retry_draft, domain=domain, verdict=retry_verdict_word,
            )
            phase_artifact_ids["retry"] = retry_artifact_id

        best_specialist, best_specialist_word, best_specialist_line = pick_better_candidate(
            draft,
            draft_verdict_word,
            draft_verdict_line,
            retry_draft,
            retry_verdict_word,
            retry_verdict_line,
        )
        chosen = "retry" if best_specialist is retry_draft else "draft"
        best_specialist_artifact_id = retry_artifact_id if chosen == "retry" else draft_artifact_id
        cprint(f"[TIER2] Kept better specialist candidate: {chosen} ({best_specialist_word})", "dim")

    if not is_synthetic and mgr.specialist_path():
        debug_log(
            "specialist.keep_resident",
            model=Path(mgr.specialist_path()).name,
            reason="keep_alive_v1.5",
        )
        # mgr.unload_specialist()  # Disabled to keep Ollama models resident

    # Step 6 — Supervisor synthesis and keep-better selection
    result = best_specialist
    result_word = best_specialist_word
    result_line = best_specialist_line
    result_artifact_id = best_specialist_artifact_id
    if ENABLE_TIER2_SYNTHESIS and best_specialist_word != "PASS":
        synthesis_system = (
            "You are a brilliant coordinator. A domain specialist produced the following draft. "
            "Refine it into a precise, complete, well-structured final answer."
        )
        specialist_for_synthesis = (
            session_store.get_generation_text(best_specialist_artifact_id)
            if (session_store and best_specialist_artifact_id)
            else best_specialist
        )
        # Skeleton = full prompt with draft absent; measures tokens available for draft.
        _synth_skeleton = (
            f"Original question: {query}\n\nSpecialist draft:\n\nProvide the refined final answer:"
        )
        draft_fitted = fit_draft(
            specialist_for_synthesis, _synth_skeleton, mgr,
            SUPERVISOR_OUTPUT_BUDGET_T2, THINK_BUDGET_T2,
            system=synthesis_system,
            max_draft_tokens=SYNTHESIS_DRAFT_TOKEN_CAP,
        )
        synthesis_prompt = (
            f"Original question: {query}\n\nSpecialist draft:\n{draft_fitted}\n\nProvide the refined final answer:"
        )
        if show_intermediate:
            cprint("\nResponse (Supervisor Synthesis)", "bold")
        synth_result = mgr.generate_supervisor(
            synthesis_prompt, system=synthesis_system,
            max_tokens=SUPERVISOR_OUTPUT_BUDGET_T2, temperature=0.2,
            think_budget=THINK_BUDGET_T2,
            on_token=_stream_token if show_intermediate else None,
        )
        if domain == "code":
            synth_result = normalize_python_function_output(query, synth_result)
        if show_intermediate:
            print()  # newline after streaming

        synth_verdict_word, synth_verdict_line = grade_candidate(
            query, synth_result, domain, mgr, grade_system,
            self_grade=self_grade,
            use_specialist_self_grade=((not is_synthetic) and domain in SPECIALIST_SELF_GRADE_DOMAINS),
        )
        cprint(f"[Grade synthesis] {synth_verdict_line}", "dim")

        synth_artifact_id = ""
        if session_store:
            synth_artifact_id = session_store.add_generation_artifact(
                task_key, "tier2_synthesis", synth_result, domain=domain, verdict=synth_verdict_word,
            )
            phase_artifact_ids["synthesis"] = synth_artifact_id

        result, result_word, result_line = pick_better_candidate(
            best_specialist,
            best_specialist_word,
            best_specialist_line,
            synth_result,
            synth_verdict_word,
            synth_verdict_line,
        )
        source = "synthesis" if result is synth_result else "specialist"
        result_artifact_id = synth_artifact_id if source == "synthesis" else best_specialist_artifact_id
        cprint(f"[TIER2] Final source: {source} ({result_word})", "dim")
    elif ENABLE_TIER2_SYNTHESIS:
        cprint("[TIER2] Synthesis bypassed — specialist candidate already PASS.", "dim")
    else:
        cprint("[TIER2] Synthesis bypassed — returning best specialist candidate.", "dim")

    final_word, final_line = result_word, result_line
    cprint(f"[Grade final] {final_line}", "dim")

    if should_trigger_supervisor_fallback(final_word, domain):
        cprint("[TIER2] Score-gated fallback: supervisor direct pass", "yellow")
        cprint("\nResponse (Supervisor Fallback)", "bold")
        fallback = run_supervisor_fallback(
            query=query,
            mgr=mgr,
            failure_summary=final_line,
            max_tokens=SUPERVISOR_OUTPUT_BUDGET_T2,
            think_budget=THINK_BUDGET_T2,
            on_token=_stream_token,
        )
        print()  # newline after streaming
        if domain == "code":
            fallback = normalize_python_function_output(query, fallback)
        # render_response skipped — fallback was streamed live above

        fb_word, fb_line = grade_candidate(
            query, fallback, domain, mgr, grade_system,
            self_grade=self_grade,
            use_specialist_self_grade=((not is_synthetic) and domain in SPECIALIST_SELF_GRADE_DOMAINS),
        )
        cprint(f"[Grade fallback] {fb_line}", "dim")
        fallback_artifact_id = ""
        if session_store:
            fallback_artifact_id = session_store.add_generation_artifact(
                task_key, "tier2_fallback", fallback, domain=domain, verdict=fb_word,
            )
            phase_artifact_ids["fallback"] = fallback_artifact_id
        result, result_word, _ = pick_better_candidate(
            result,
            final_word,
            final_line,
            fallback,
            fb_word,
            fb_line,
        )
        picked = "fallback" if result is fallback else "pipeline"
        if picked == "fallback":
            result_artifact_id = fallback_artifact_id
        cprint(f"[TIER2] Post-fallback source: {picked} ({result_word})", "dim")

    # Persist
    if session_store:
        artifacts = {}
        if match:
            artifacts["template"] = match.template_id
        artifacts["generation_artifact_ids"] = phase_artifact_ids
        artifacts["final_artifact_id"] = result_artifact_id
        tier = "TIER2_SYNTHETIC" if is_synthetic else "TIER2"
        session_store.write_task(query, tier, [domain], result, artifacts)

    return result


# ═══════════════════════════════════════════════════════════════════════════




