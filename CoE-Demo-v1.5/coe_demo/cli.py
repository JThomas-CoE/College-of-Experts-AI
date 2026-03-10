
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
from .pipeline import *
from .memory import *
from .templates import *
from .kb import *
from .skills import *
from .tier3_stub import run_tier3


# PHASE 8 — CLI Shell
# ═══════════════════════════════════════════════════════════════════════════


def settings_table(settings: dict, sources: dict) -> None:
    if RICH_AVAILABLE and console:
        tbl = Table(title="[bold]College of Experts — Settings[/bold]", show_lines=True)
        tbl.add_column("Setting", style="cyan")
        tbl.add_column("Value")
        tbl.add_column("Source", style="dim")
        tbl.add_column("Mutable", style="dim")
        mutability = {
            "device":  "restart-only",
            "session": "/new to rotate",
            "enhance": "/enhance on|off",
            "confirm": "/confirm on|off",
            "kb":      "/kb on|off",
            "self_grade": "/selfgrade on|off",
            "render_mode": "/render intermediate|final",
            "embed_cpu_batch_size": "restart-only",
            "embed_cpu_threads": "restart-only",
            "max_runtime_seq_tokens": "restart-only",
        }
        for k in ["device", "session", "enhance", "confirm", "kb", "self_grade", "render_mode", "embed_cpu_batch_size", "embed_cpu_threads", "max_runtime_seq_tokens"]:
            v = settings.get(k)
            display = str(v) if v is not None else "(auto)"
            tbl.add_row(k, display, sources.get(k, "default"), mutability.get(k, ""))
        console.print(tbl)
    else:
        print(f"\n{'Setting':<12} {'Value':<20} {'Source':<16} {'Mutable'}")
        print("─" * 64)
        for k in ["device", "session", "enhance", "confirm", "kb", "self_grade", "render_mode", "embed_cpu_batch_size", "embed_cpu_threads", "max_runtime_seq_tokens"]:
            v = settings.get(k)
            display = str(v) if v is not None else "(auto)"
            src = sources.get(k, "default")
            mut = {
                "device": "restart-only",
                "session": "/new to rotate",
                "enhance": "/enhance on|off",
                "confirm": "/confirm on|off",
                "kb": "/kb on|off",
                "self_grade": "/selfgrade on|off",
                "render_mode": "/render intermediate|final",
                "embed_cpu_batch_size": "restart-only",
                "embed_cpu_threads": "restart-only",
                "max_runtime_seq_tokens": "restart-only",
            }.get(k, "")
            print(f"{k:<12} {display:<20} {src:<16} {mut}")
        print()


def domains_table(extra_dirs: Optional[List[str]] = None) -> None:
    if RICH_AVAILABLE and console:
        tbl = Table(title="[bold]Domain → Model[/bold]", show_lines=True)
        tbl.add_column("Domain", style="cyan")
        tbl.add_column("Model Path")
        for domain, name in DOMAIN_MODELS.items():
            path = resolve_model_path(name, extra_dirs)
            exists = "✓" if Path(path).exists() else "✗ MISSING"
            tbl.add_row(domain, f"[{'green' if '✓' in exists else 'red'}]{path} {exists}[/]")
        console.print(tbl)
    else:
        print(f"\n{'Domain':<12}  {'Model Path'}")
        print("─" * 60)
        for domain, path in DOMAIN_MODELS.items():
            exists = "✓" if Path(path).exists() else "✗ MISSING"
            print(f"{domain:<12}  {path}  {exists}")
        print()


def dispatch(
    query: str,
    mgr: ModelManager,
    session_store: SessionStore,
    template_store: TemplateStore,
    skill_store: "SkillStore",
    settings: dict,
) -> None:
    """Classify and dispatch a query through the appropriate pipeline."""
    enhance = settings.get("enhance", True)
    kb_on = settings.get("kb", True)
    confirm_on = settings.get("confirm", True)
    loop_guard = settings.get("loop_guard", True)
    self_grade = is_self_grading_enabled(settings)
    render_mode = normalize_render_mode(settings.get("render_mode", POLICY_V1_RENDER_DEFAULT))

    t0 = time.time()
    debug_log("dispatch.begin", query_chars=len(query or ""))

    def _is_followup_edit(text: str) -> bool:
        q = (text or "").strip().lower()
        return bool(re.match(r"^(make|modify|change|update|add|remove|tweak|improve|adjust|fix)\b", q))

    def _is_referential_artifact_followup(text: str, prev_task: dict) -> bool:
        q = f" {(text or '').strip().lower()} "
        prev_domains_local = [str(d).lower() for d in (prev_task.get("domains", []) or []) if d]
        if len(prev_domains_local) != 1 or prev_domains_local == ["supervisor"]:
            return False

        prev_artifacts = prev_task.get("artifacts", {}) or {}
        has_prior_output = bool(
            prev_task.get("output_snippet")
            or (isinstance(prev_artifacts, dict) and prev_artifacts.get("final_artifact_id"))
            or (isinstance(prev_artifacts, dict) and prev_artifacts.get("generation_artifact_ids"))
        )
        if not has_prior_output:
            return False

        referential_markers = (
            " it ", " it's ", " it’s ", " they ", " them ", " this ", " that ",
            " these ", " those ", " the app ", " the page ", " the site ",
            " the code ", " the function ", " the script ", " the query ",
            " the output ", " the response ", " you just built ", " you built ",
            " you just wrote ", " you wrote ", " generated ", " above ", " previous ",
        )
        issue_or_edit_markers = (
            "bug", "issue", "problem", "broken", "breaks", "fails", "failing",
            "not working", "wrong", "error", "disappear", "disappears", "missing",
            "accumulate", "crash", "fix", "update", "change", "modify", "improve",
            "add", "remove", "when i ", " if i ", " sorry but", " actually",
        )

        has_reference = any(marker in q for marker in referential_markers)
        has_issue_or_edit = any(marker in q for marker in issue_or_edit_markers)
        return has_reference and has_issue_or_edit

    def _has_explicit_code_intent(text: str) -> bool:
        q = (text or "").lower()
        return bool(re.search(r"\b(python|javascript|typescript|function|class|method|algorithm|script|refactor|debug|compile|module)\b", q))

    def _parse_routing_correction(text: str) -> Optional[Tuple[str, List[str]]]:
        raw = (text or "").strip().lower()
        if not raw:
            return None

        _all_domains = ("code", "sql", "math", "medical", "legal", "web", "supervisor")

        def _extract_domains(s: str) -> List[str]:
            found: List[str] = []
            for dom in _all_domains:
                if re.search(r"(?<![a-zA-Z0-9])" + re.escape(dom) + r"(?![a-zA-Z0-9])", s):
                    found.append(dom)
            return found

        tier_match = re.search(r"\btier\s*([123])\b", raw)

        if not tier_match:
            # Accept domain-only corrections: a short phrase or one containing routing
            # language, e.g. "web", "code", "routing should be web", "use code domain".
            routing_words = ("routing", "route", "should be", "use ", "domain", "tier")
            found_domains = _extract_domains(raw)
            is_routing_hint = len(raw.split()) <= 4 or any(p in raw for p in routing_words)
            if found_domains and is_routing_hint:
                return "TIER2", found_domains[:1]
            return None

        tier = f"TIER{tier_match.group(1)}"
        domains: List[str] = _extract_domains(raw)
        if not domains:
            domains = ["supervisor"] if tier == "TIER1" else ["web"]
        if tier == "TIER3" and len(domains) < 2:
            tier = "TIER2"
        if tier == "TIER2" and len(domains) >= 2:
            tier = "TIER3"
        return tier, domains

    try:
        tier, domains, interpretation = classify_query(query, mgr, confirm_enabled=confirm_on)
        debug_log("dispatch.classified", tier=tier, domains=domains, interpretation=interpretation[:200])

        recent = session_store.get_recent_tasks(1) if session_store else []
        prev = recent[0] if recent else {}
        prev_domains = [str(d).lower() for d in (prev.get("domains", []) or []) if d]
        prev_tier = str(prev.get("tier", "TIER2")).upper()
        followup = _is_followup_edit(query)
        referential_followup = _is_referential_artifact_followup(query, prev)

        if (followup or referential_followup) and prev_domains:
            if tier == "TIER1":
                tier = "TIER2"
                domains = prev_domains
                cprint(f"[RoutingGuard] Follow-up continuity override → {tier} / {domains}", "dim")
                debug_log("dispatch.routing_override", reason="followup_from_tier1", tier=tier, domains=domains)
            elif (followup or referential_followup) and len(prev_domains) == 1 and prev_domains != ["supervisor"]:
                tier = "TIER2"
                domains = prev_domains
                cprint(f"[RoutingGuard] Follow-up continuity override → {tier} / {domains}", "dim")
                debug_log("dispatch.routing_override", reason="followup_continuity", tier=tier, domains=domains)
            elif prev_domains == ["web"] and domains == ["code"] and not _has_explicit_code_intent(query):
                tier = "TIER2"
                domains = ["web"]
                cprint("[RoutingGuard] Follow-up continuity override → TIER2 / ['web']", "dim")
                debug_log("dispatch.routing_override", reason="followup_web_over_code", tier=tier, domains=domains)

        # Typo-tolerant follow-up guard: if Stage 2 routed to supervisor but the prior
        # task was a specialist domain and the query looks like an edit/change request
        # (even with a typo like "changer" instead of "change"), inherit the prior domain.
        _EDIT_LIKE = re.compile(
            r"^(make|modif|chang|updat|add|remov|tweak|improv|adjust|fix|set|increas|decreas|replac|renam|restyle|resize|recolor|reformat)",
            re.IGNORECASE,
        )
        if (
            domains == ["supervisor"]
            and len(prev_domains) == 1
            and prev_domains != ["supervisor"]
            and _EDIT_LIKE.match(query.strip())
        ):
            tier = "TIER2"
            domains = prev_domains
            cprint(f"[RoutingGuard] Typo-tolerant follow-up override → {tier} / {domains}", "dim")
            debug_log("dispatch.routing_override", reason="typo_tolerant_followup", tier=tier, domains=domains)

    except Exception as e:
        cprint(f"[ERROR] Classification failed: {e}", "red")
        debug_log("dispatch.classify_error", error=str(e))
        return

    if tier == "TIER1" or not confirm_on:
        # TIER1 is trivial — skip confirmation and route directly.
        cprint(f"[CoE] Routing: {tier} / {domains}", "dim")
    else:
        # TIER2 / TIER3 — show interpretation and allow correction.
        cprint(f"\n[CoE] I interpret this as: {interpretation}")
        cprint(f"  → Routing: {tier} / {domains}")
        try:
            correction = input("  [Enter to proceed, or type a correction] ").strip()
        except (EOFError, KeyboardInterrupt):
            correction = ""
        if correction:
            cprint(f"[CoE] Re-routing with correction: {correction}", "dim")
            parsed = _parse_routing_correction(correction)
            if parsed is not None:
                tier, domains = parsed
                interpretation = f"User routing override: {tier} / {domains}"
                cprint(f"[CoE] Applied routing override → {tier} / {domains}", "dim")
                debug_log("dispatch.routing_override", reason="user_correction", tier=tier, domains=domains)
            else:
                # Treat correction as query rewrite only when no explicit routing override is detected.
                query = correction
                try:
                    tier, domains, interpretation = classify_query(query, mgr, confirm_enabled=False)
                except Exception as e:
                    cprint(f"[ERROR] Re-classification failed: {e}", "red")
                    return

    try:
        domains = [d for d in domains if d in DOMAIN_MODELS] or ["supervisor"]
        if tier != "TIER1" and len(domains) >= 2:
            tier = "TIER3"
        elif tier == "TIER3" and len(domains) < 2:
            tier = "TIER2"

        if tier == "TIER1":
            debug_log("dispatch.route", tier="TIER1", domains=domains)
            output = run_tier1(query, mgr)
        elif tier == "TIER3":
            debug_log("dispatch.route", tier="TIER3", domains=domains)
            output = run_tier3(query, domains, mgr, session_store, template_store)
        else:
            domain = domains[0] if domains else "supervisor"
            debug_log("dispatch.route", tier="TIER2", domains=[domain])
            output = run_tier2(
                query, domain, mgr, session_store, template_store,
                skill_store=skill_store,
                enhance=enhance, kb_enabled=kb_on, self_grade=self_grade, render_mode=render_mode,
                resume_after_switch=False,
                router_followup=(followup or referential_followup),
                loop_guard=loop_guard,
            )
    except Exception as e:
        cprint(f"[ERROR] Pipeline error: {e}", "red")
        debug_log("dispatch.pipeline_error", error=str(e), tier=tier, domains=domains)
        return

    if not (output or "").strip():
        cprint("[CoE] Empty pipeline output — returning friendly fallback message.", "yellow")
        debug_log("dispatch.empty_output", tier=tier, domains=domains)
        output = EMPTY_RESPONSE_MESSAGE

    if tier == "TIER1" and session_store:
        task_key = session_store.peek_task_key()
        final_artifact_id = session_store.add_generation_artifact(
            task_key,
            "tier1_final",
            output,
            domain="supervisor",
            verdict="PASS",
        )
        session_store.write_task(
            query,
            "TIER1",
            ["supervisor"],
            output,
            {
                "generation_artifact_ids": {"final": final_artifact_id},
                "final_artifact_id": final_artifact_id,
            },
        )

    if session_store:
        try:
            session_store.maybe_archive_history(mgr, mgr._embedder)
        except Exception as mem_err:
            cprint(f"[Memory] archive warning: {mem_err}", "dim")
            debug_log("dispatch.memory_archive_warning", error=str(mem_err))

    try:
        mgr.reconcile_residency(None, reason=f"post-{tier}")
    except Exception as audit_err:
        cprint(f"[Residency] audit warning: {audit_err}", "yellow")
        debug_log("dispatch.residency_warning", error=str(audit_err))

    elapsed = time.time() - t0

    # TIER1 and TIER2 output is already streamed live — skip the panel re-render.
    # TIER3 (stub) and any non-streaming path still display via render_response.
    if tier not in ("TIER1", "TIER2", "TIER2_SYNTHETIC"):
        try:
            render_response(output, title="Response")
        except Exception as print_err:
            # Fallback: plain print so the answer is never silently lost
            try:
                print("\n" + "─" * 60)
                print(output)
                print("─" * 60)
            except Exception:
                pass
            cprint(f"[WARN] Rich render failed ({print_err}); fell back to plain print.", "yellow")
            debug_log("dispatch.render_warning", error=str(print_err))

    debug_log("dispatch.end", tier=tier, domains=domains, elapsed_s=round(elapsed, 3), output_chars=len(output or ""))
    cprint(f"\n⏱  {elapsed:.1f}s  |  {tier} / {domains}", "dim")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="College of Experts — Interactive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ollamaDemo.py\n"
            "  python ollamaDemo.py --device cpu --no-kb\n"
            "  python ollamaDemo.py --no-confirm --session myproject\n"
        ),
    )
    parser.add_argument("--device", default="dml",
                        choices=["dml", "cpu", "cuda"],
                        help="Compute device (default: dml — DirectML iGPU Radeon 890M)")
    parser.add_argument("--no-kb", action="store_true",
                        help="Disable KB retrieval")
    parser.add_argument("--no-enhance", action="store_true",
                        help="Disable supervisor prompt enhancement (raw passthrough)")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Disable interpretation confirmation prompt")
    parser.add_argument("--no-self-grade", action="store_true",
                        help="Disable model-based self-grading")
    parser.add_argument("--session", default=None,
                        help="Override session ID prefix")
    parser.add_argument("--embed-cpu-batch-size", type=int, default=None,
                        help="CPU embedding batch size for first-run template indexing")
    parser.add_argument("--embed-cpu-threads", type=int, default=None,
                        help="CPU threads for embedding model when running on CPU")
    parser.add_argument("--debug-trace", action="store_true",
                        help="Enable JSONL breadcrumb trace at acceptance_runs/debug_trace.jsonl")
    args = parser.parse_args()

    debug_from_env = os.getenv("COE_DEBUG_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}
    debug_enabled = bool(args.debug_trace or debug_from_env)
    set_debug_trace(debug_enabled)
    if debug_enabled:
        cprint(f"[DEBUG] Trace enabled: {DEBUG_TRACE_PATH}", "dim")
        debug_log("process.start", argv=" ".join(sys.argv))

    # Load settings
    settings, sources = load_settings(args)

    # Startup banner
    enhance_str = "on" if settings["enhance"] else "off"
    confirm_str = "on" if settings["confirm"] else "off"
    kb_str      = "on" if settings["kb"] else "off"
    sess_str    = settings["session"] or "(auto)"

    banner_text = (
        f"[bold cyan]College of Experts — Interactive CLI v{VERSION}[/bold cyan]\n"
        f"device=[yellow]{settings['device']}[/yellow]  "
        f"enhance=[yellow]{enhance_str}[/yellow]  "
        f"confirm=[yellow]{confirm_str}[/yellow]  "
        f"kb=[yellow]{kb_str}[/yellow]  "
        f"session=[yellow]{sess_str}[/yellow]\n"
        f"Type [bold]/help[/bold] for commands."
        if RICH_AVAILABLE else
        f"College of Experts — Interactive CLI v{VERSION}\n"
        f"device={settings['device']}  enhance={enhance_str}  confirm={confirm_str}  kb={kb_str}\n"
        f"Type /help for commands."
    )
    if RICH_AVAILABLE and console:
        console.print(Panel(banner_text, border_style="cyan"))
    else:
        print(banner_text)

    # Instantiate core objects
    mgr = ModelManager(
        device=settings["device"],
        model_base_dir=settings.get("model_base_dir"),
        embed_cpu_batch_size=settings.get("embed_cpu_batch_size", 96),
        embed_cpu_threads=settings.get("embed_cpu_threads", 12),
        max_runtime_seq_tokens=settings.get("max_runtime_seq_tokens", 16384),
    )
    session_store = SessionStore(session_prefix=settings.get("session"))
    template_store = TemplateStore()
    skill_store    = SkillStore()

    import atexit
    atexit.register(mgr.shutdown)

    try:
        mgr.startup()
    except FileNotFoundError as e:
        cprint(f"[FATAL] {e}", "red")
        sys.exit(1)
    except Exception as e:
        cprint(f"[FATAL] ModelManager startup failed: {e}", "red")
        sys.exit(1)

    # Init KB with embedder if available
    if mgr._embedder:
        init_kb(mgr._embedder)

    # Pre-compute template and skill embeddings
    if mgr._embedder:
        template_store.startup(mgr._embedder)
        skill_store.startup(mgr._embedder)
        try:
            mgr._embedder.compact()
        except Exception:
            pass

    cprint(f"[SessionStore] Session: {session_store.session_id}", "dim")
    debug_log("session.started", session_id=session_store.session_id)
    _extra = [settings.get("model_base_dir")] if settings.get("model_base_dir") else None
    domains_table(_extra)

    HELP_TEXT = """
[bold]Commands:[/bold]
  /quit, /exit       — Exit the shell
  /new               — Start a new session
  /status            — Show model and session status
  /memory [N]        — Show last N tasks from session (default 5)
  /domains           — Show domain → model table
  /enhance [on|off]  — Toggle auto prompt enhancement
  /confirm [on|off]  — Toggle interpretation confirmation
  /kb [on|off]       — Toggle KB retrieval
    /selfgrade [on|off] — Toggle model-based self-grading
    /render [intermediate|final] — Toggle intermediate response panels
  /config            — Show all settings with source and mutability
  /config set KEY V  — Persist a mutable setting (writes config/demo_config.json)
  /help              — Show this message
  <anything else>    — Classify and dispatch as a query

[bold]Query flags (append anywhere in your prompt):[/bold]
  --noloop           — Disable loop-guard for this query (use when the output
                       will contain legitimately repetitive data: sparse matrices,
                       large lookup tables, board representations, etc.)
""" if RICH_AVAILABLE else """
Commands:
  /quit /exit         Exit
  /new                New session
  /status             Model + session status
  /memory [N]         Last N tasks (default 5)
  /domains            Domain table
  /enhance [on|off]   Toggle enhancement
  /confirm [on|off]   Toggle confirmation
  /kb [on|off]        Toggle KB
    /selfgrade [on|off] Toggle model-based self-grading
    /render [intermediate|final] Toggle intermediate panels
  /config             Show all settings
  /config set K V     Persist a setting
  /help               This message

Query flags:
  --noloop            Disable loop-guard for this query (sparse matrices,
                      lookup tables, board representations, etc.)
"""

    # Main loop
    while True:
        try:
            user_input = input("\n[CoE] > ").strip()
        except (EOFError, KeyboardInterrupt):
            cprint("\n[CoE] Goodbye.", "dim")
            mgr.shutdown()
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # ── Shell commands ────────────────────────────────────────────────

        if cmd in ("/quit", "/exit"):
            cprint("[CoE] Goodbye. Shutting down models…", "dim")
            debug_log("shell.exit")
            mgr.shutdown()
            break

        elif cmd == "/new":
            session_store.clear_session()
            cprint(f"[CoE] New session: {session_store.session_id}")

        elif cmd == "/status":
            sv = mgr.supervisor_path()
            sp = mgr.specialist_path()
            stats = session_store.get_stats()
            cprint(f"  supervisor : {sv or '(not loaded)'}")
            cprint(f"  specialist : {sp or '(none)'}")
            cprint(f"  load_thread: {'active' if (mgr._load_thread and mgr._load_thread.is_alive()) else 'idle'}")
            cprint(f"  pending    : {mgr._pending_specialist_path or '(none)'}")
            cprint(f"  embedder   : {getattr(mgr._embedder, 'device', '(disabled)')}")
            cprint(f"  enhance    : {settings['enhance']}")
            cprint(f"  kb         : {settings['kb']}")
            cprint(f"  confirm    : {settings['confirm']}")
            cprint(f"  self_grade : {is_self_grading_enabled(settings)}")
            cprint(f"  render     : {normalize_render_mode(settings.get('render_mode'))}")
            cprint(f"  emb_batch  : {settings.get('embed_cpu_batch_size')}")
            cprint(f"  emb_threads: {settings.get('embed_cpu_threads')}")
            cprint(f"  seq_cap    : {settings.get('max_runtime_seq_tokens')}")
            cprint(f"  session    : {stats['session_id']}  tasks={stats['task_count']}")
            cprint(f"  artifacts  : {stats.get('artifact_count', 0)}")

        elif cmd.startswith("/memory"):
            parts = user_input.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
            tasks = session_store.get_recent_tasks(n)
            if not tasks:
                cprint("[CoE] No tasks in current session.", "dim")
            else:
                for t in tasks:
                    tier = t.get("tier", "?")
                    doms = ",".join(t.get("domains", []))
                    q = t.get("query", t.get("query_snippet", ""))[:80]
                    cprint(f"  [{tier} | {doms}] {q}")

        elif cmd == "/domains":
            domains_table([settings.get("model_base_dir")] if settings.get("model_base_dir") else None)

        elif cmd.startswith("/enhance"):
            parts = user_input.split()
            if len(parts) > 1:
                val = parts[1].lower() == "on"
                settings["enhance"] = val
                sources["enhance"] = "runtime"
                save_setting("enhance", val)
                cprint(f"[CoE] enhance = {'on' if val else 'off'}")
            else:
                cprint(f"[CoE] enhance = {'on' if settings['enhance'] else 'off'}")

        elif cmd.startswith("/confirm"):
            parts = user_input.split()
            if len(parts) > 1:
                val = parts[1].lower() == "on"
                settings["confirm"] = val
                sources["confirm"] = "runtime"
                save_setting("confirm", val)
                cprint(f"[CoE] confirm = {'on' if val else 'off'}")
            else:
                cprint(f"[CoE] confirm = {'on' if settings['confirm'] else 'off'}")

        elif cmd.startswith("/kb"):
            parts = user_input.split()
            if len(parts) > 1:
                val = parts[1].lower() == "on"
                settings["kb"] = val
                sources["kb"] = "runtime"
                save_setting("kb", val)
                cprint(f"[CoE] kb = {'on' if val else 'off'}")
            else:
                cprint(f"[CoE] kb = {'on' if settings['kb'] else 'off'}")

        elif cmd.startswith("/selfgrade"):
            parts = user_input.split()
            if len(parts) > 1:
                val = parts[1].lower() == "on"
                settings["self_grade"] = val
                sources["self_grade"] = "runtime"
                save_setting("self_grade", val)
                cprint(f"[CoE] self_grade = {'on' if val else 'off'}")
            else:
                cprint(f"[CoE] self_grade = {'on' if is_self_grading_enabled(settings) else 'off'}")

        elif cmd.startswith("/render"):
            parts = user_input.split()
            if len(parts) > 1:
                mode_token = parts[1].lower()
                if mode_token in ("final", "final-only", "final_only"):
                    mode = "final only"
                elif mode_token in ("intermediate", "both", "all", "default"):
                    mode = "intermediate+final"
                else:
                    cprint("[CoE] render usage: /render intermediate|final", "yellow")
                    continue
                settings["render_mode"] = mode
                sources["render_mode"] = "runtime"
                save_setting("render_mode", mode)
                cprint(f"[CoE] render_mode = {mode}")
            else:
                cprint(f"[CoE] render_mode = {normalize_render_mode(settings.get('render_mode'))}")

        elif cmd.startswith("/config set"):
            parts = user_input.split()
            if len(parts) < 4:
                cprint("[CoE] Usage: /config set KEY VALUE", "yellow")
            else:
                key = parts[2].lower()
                val_str = parts[3]
                if key == "device":
                    cprint("[CoE] 'device' is restart-only. Change in config/demo_config.json and restart.", "yellow")
                elif key in ("enhance", "confirm", "kb", "self_grade"):
                    val = val_str.lower() in ("on", "true", "1", "yes")
                    settings[key] = val
                    sources[key] = "runtime"
                    save_setting(key, val)
                    cprint(f"[CoE] {key} = {val}  (saved to config)")
                elif key == "render_mode":
                    mode = normalize_render_mode(val_str)
                    settings["render_mode"] = mode
                    sources["render_mode"] = "runtime"
                    save_setting("render_mode", mode)
                    cprint(f"[CoE] render_mode = {mode}  (saved to config)")
                elif key in ("embed_cpu_batch_size", "embed_cpu_threads", "max_runtime_seq_tokens"):
                    try:
                        val = int(val_str)
                        if val < 1:
                            raise ValueError()
                    except ValueError:
                        cprint(f"[CoE] {key} must be a positive integer", "yellow")
                        continue
                    settings[key] = val
                    sources[key] = "runtime"
                    save_setting(key, val)
                    cprint(f"[CoE] {key} = {val}  (saved to config; restart to apply)")
                elif key == "session":
                    settings["session"] = val_str
                    sources["session"] = "runtime"
                    save_setting("session", val_str)
                    cprint(f"[CoE] session prefix = {val_str}  (saved to config)")
                else:
                    cprint(f"[CoE] Unknown setting '{key}'. Valid: device, enhance, confirm, kb, self_grade, render_mode, embed_cpu_batch_size, embed_cpu_threads, max_runtime_seq_tokens, session", "yellow")

        elif cmd.startswith("/config"):
            settings_table(settings, sources)

        elif cmd == "/help":
            if RICH_AVAILABLE and console:
                console.print(Markdown(HELP_TEXT))
            else:
                print(HELP_TEXT)

        # ── Query dispatch ────────────────────────────────────────────────
        else:
            try:
                # Allow --noloop anywhere in the prompt to disable the loop-guard for this
                # query only.  Useful for queries that produce legitimately repetitive output
                # such as sparse matrices, large lookup tables, or other data-heavy structures.
                _query_text = user_input
                _dispatch_settings = settings
                if re.search(r'--noloop\b', _query_text, re.IGNORECASE):
                    _query_text = re.sub(r'\s*--noloop\b', '', _query_text, flags=re.IGNORECASE).strip()
                    _dispatch_settings = {**settings, "loop_guard": False}
                    cprint("[CoE] Loop guard disabled for this query.", "yellow")
                dispatch(_query_text, mgr, session_store, template_store, skill_store, _dispatch_settings)
            except KeyboardInterrupt:
                cprint("\n[CoE] Request cancelled by user. (Type /quit to exit the app)", "yellow")
                if mgr.specialist_path():
                    # Aggressively flush Ollama VRAM to abort generating the rest of the stream
                    import urllib.request, json
                    model_name = mgr.specialist_path().replace('ollama://', '')
                    try:
                        urllib.request.urlopen(
                            urllib.request.Request("http://localhost:11434/api/generate",
                                data=json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8"),
                                headers={"Content-Type": "application/json"}),
                            timeout=45
                        )
                    except Exception:
                        pass
                    mgr.unload_specialist()
            except Exception as dispatch_err:
                cprint(f"[ERROR] Unhandled exception in pipeline: {dispatch_err}", "red")
                debug_log("dispatch.unhandled_exception", error=str(dispatch_err))
                import traceback
                traceback.print_exc()
                # Ensure specialist is freed so the next query starts clean
                if mgr.specialist_path():
                    mgr.unload_specialist()


if __name__ == "__main__":
    main()




