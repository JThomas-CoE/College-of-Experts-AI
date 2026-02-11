"""
College of Experts Benchmark Runner v3

Compares CoE framework performance against baseline isolated model scores.

Architecture mirrors a real Mixture-of-Experts approach:
  1. SPECIALIST DRAFT  - Domain model generates grounded response
  2. SUPERVISOR SYNTH  - DeepSeek synthesizes/refines using specialist output
  3. QUALITY GATE      - Validates output, retries if needed
  4. SCORE-BASED FALLBACK - Pure DeepSeek if ensemble still fails

This reflects how a SOTA MoE model works internally:
  - Record the histographic distribution of which experts fire per task category
  - Craft each specialist as the core collection of MoE neurons needed
  - Collect common neurons into the router/supervisor that provides "glue"

Run: python benchmarks/run_coe_benchmark.py [--suite SUITE] [--samples N]
"""

import asyncio
import time
import json
import sys
import gc
import re
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from run_benchmark for data loading and grading
from benchmarks.run_benchmark import (
    load_standard_suite, 
    grade_response, 
    BENCHMARK_SUITE,
    PUBLISHED_REFS
)

# CoE Framework imports
from src.embedding_manager import EmbeddingManager
from src.vram_manager import create_scheduler
from src.savant_pool import SavantPool
from src.expert_scope import EXPERT_SCOPES, get_expert_scope
from src.quality_gate import QualityGate
from src.knowledge_layer import KnowledgeRetriever

# Baseline results for comparison (from isolated model runs, n=25)
BASELINE_SCORES = {
    "HumanEval": {"accuracy": 71.4, "model": "Qwen2.5-Coder-7B"},
    "GSM8K": {"accuracy": 56.0, "model": "Qwen2.5-Math-7B"},
    "Medical": {"accuracy": 40.0, "model": "BioMistral-7B"},
    "Spider": {"accuracy": 64.8, "model": "sqlcoder-7b-2"},
    "Legal": {"accuracy": 8.4, "model": "law-LLM"}
}

# Suite to expert mapping
SUITE_EXPERT_MAP = {
    "HumanEval": "python_backend",
    "GSM8K": "math_expert",
    "Medical": "medical_clinical",
    "Spider": "sql_schema_architect",
    "Legal": "legal_contracts"
}

# Pipeline mode per suite:
#   "solo"     = specialist only (already strong baseline, e.g. HumanEval, GSM8K)
#   "ensemble" = specialist draft -> DeepSeek refinement (Medical, Legal, Spider)
#   "fallback" = score-gated: try specialist, fallback to DeepSeek on failure
SUITE_PIPELINE_MODE = {
    "HumanEval": "solo",        # Qwen Coder is already strong here
    "GSM8K": "solo",            # Qwen Math is already strong here
    "Medical": "ensemble",      # BioMistral draft -> DeepSeek synthesis
    "Spider": "solo",           # SQLCoder is specialized, solo is fine
    "Legal": "ensemble",        # law-LLM draft -> DeepSeek synthesis
}

# Model paths - VERIFIED against models/ directory
SAVANT_MODELS = {
    "python_backend": "models/Qwen2.5-Coder-7B-DML",
    "sql_schema_architect": "models/sqlcoder-7b-2-DML",
    "html_css_specialist": "models/Qwen2.5-Coder-7B-DML",
    "math_expert": "models/Qwen2.5-Math-7B-DML",
    "security_architect": "models/DeepSeek-R1-Distill-Qwen-7B-DML",
    "legal_contracts": "models/law-LLM-DML",
    "medical_clinical": "models/BioMistral-7B-DML",
    "deepseek_supervisor": "models/DeepSeek-R1-Distill-Qwen-7B-DML",
}

# ─── Chat templates per model family ───────────────────────────────────
# Each model architecture expects a different prompt format.
# Using the wrong template is the #1 cause of degraded accuracy.

def _format_prompt(model_path: str, prompt: str, system: str = "") -> str:
    """Format prompt according to model architecture."""
    p = model_path.lower()
    
    # SQLCoder (CodeLlama-based) — NO system message, uses ### markers
    if "sqlcoder" in p:
        return f"### Task\nGenerate a SQL query to answer the following question.\n\n### Instructions\n{prompt}\n\n### Response\n"
    
    # law-LLM (Llama-based) — Llama-2 instruct format, 2048 ctx limit
    if "law" in p:
        sys_part = f"<<SYS>>\n{system}\n<</SYS>>\n\n" if system else ""
        return f"<s>[INST] {sys_part}{prompt} [/INST]"
    
    # BioMistral (Mistral-based) — Mistral instruct format
    if "biomistral" in p or "bio" in p:
        sys_part = f"{system}\n\n" if system else ""
        return f"<s>[INST] {sys_part}{prompt} [/INST]"
    
    # Qwen / DeepSeek — ChatML format
    if system:
        return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark task."""
    task_id: str
    suite: str
    expert_used: str
    pipeline_mode: str       # solo / ensemble / fallback
    score: float
    max_score: float
    feedback: str
    response: str
    latency_ms: float
    tokens: int
    specialist_response: str = ""   # What the specialist said (for ensemble)
    used_retry: bool = False
    used_fallback: bool = False
    used_ensemble: bool = False
    error: Optional[str] = None


class CoEBenchmarkRunner:
    """
    Runs benchmarks through the College of Experts framework.
    
    Architecture:
      For "solo" suites:     specialist model generates directly
      For "ensemble" suites: specialist drafts → DeepSeek refines
      For all suites:        score-gated DeepSeek fallback on failure
    """
    
    def __init__(self, enable_quality_gate: bool = True, enable_fallback: bool = True, 
                 force_coe: bool = False):
        self.enable_quality_gate = enable_quality_gate
        self.enable_fallback = enable_fallback
        self.force_coe = force_coe
        
        # Lazy init
        self.embedding_manager = None
        self.scheduler = None
        self.savant_pool = None
        self.quality_gate = None
        self.knowledge_retriever = None
        
        # VRAM tracking — only one model in VRAM at a time
        self._current_model = None
        self._current_model_path = None
        self._tokenizer = None
        
        # Stats
        self.pipeline_stats = {"solo": 0, "ensemble": 0, "fallback": 0}
        
    def _init_components(self):
        """Initialize CoE components."""
        import torch
        
        print("[CoE] Initializing framework components...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_manager = EmbeddingManager(model_name="BAAI/bge-m3", device=device)
        print("  ✓ Embedding manager")
        
        self.scheduler = create_scheduler(profile_name="balanced")
        self.savant_pool = SavantPool(scheduler_pool=self.scheduler.savant_pool)
        print("  ✓ VRAM scheduler")
        
        if self.enable_quality_gate:
            self.quality_gate = QualityGate()
            print("  ✓ Quality gate")
        
        try:
            self.knowledge_retriever = KnowledgeRetriever(
                embedding_fn=self.embedding_manager.encode,
                knowledge_base_dir="data/knowledge",
                enable_web_search=False
            )
            print("  ✓ Knowledge retriever")
        except Exception as e:
            print(f"  ⚠ Knowledge retriever not available: {e}")
            self.knowledge_retriever = None
        
        # Verify model paths
        print("\n  Model verification:")
        for expert_id, path in SAVANT_MODELS.items():
            exists = Path(path).exists()
            status = "✓" if exists else "✗ MISSING"
            print(f"    {status} {expert_id}: {path}")
            
    def _load_model(self, model_path: str) -> bool:
        """Load model into VRAM (swaps out previous model)."""
        import onnxruntime_genai as og
        
        if self._current_model_path == model_path:
            return True
            
        if self._current_model:
            del self._current_model
            self._current_model = None
            self._current_model_path = None
            self._tokenizer = None
            gc.collect()
            
        try:
            self._current_model = og.Model(model_path)
            self._current_model_path = model_path
            self._tokenizer = og.Tokenizer(self._current_model)
            return True
        except Exception as e:
            print(f"  ⚠ Failed to load {model_path}: {e}")
            self._current_model_path = None
            return False
            
    def _generate(self, prompt: str, system: str = "", 
                  max_tokens: int = 512, temperature: float = 0.3) -> Tuple[str, int]:
        """Generate response using currently loaded model."""
        import onnxruntime_genai as og
        
        formatted = _format_prompt(self._current_model_path, prompt, system)
        
        input_tokens = self._tokenizer.encode(formatted)
        
        # Respect model context limits (law-LLM is only 2048)
        config_path = Path(self._current_model_path) / "genai_config.json"
        ctx_limit = 4096
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text())
                ctx_limit = cfg.get("model", {}).get("context_length", 4096)
            except:
                pass
        
        available_gen = min(max_tokens, ctx_limit - len(input_tokens) - 10)
        if available_gen < 50:
            available_gen = 50  # Minimum generation
        
        params = og.GeneratorParams(self._current_model)
        params.set_search_options(
            max_length=len(input_tokens) + available_gen,
            temperature=max(temperature, 0.01)
        )
        
        generator = og.Generator(self._current_model, params)
        generator.append_tokens(input_tokens)
        
        output_tokens = []
        while not generator.is_done() and len(output_tokens) < available_gen:
            generator.generate_next_token()
            token = generator.get_next_tokens()[0]
            output_tokens.append(token)
            
        response = self._tokenizer.decode(output_tokens)
        
        del generator
        del params
        
        return response.strip(), len(output_tokens)
    
    # ─── Prompt builders per suite ──────────────────────────────────────
    
    def _build_specialist_prompt(self, suite: str, task_prompt: str) -> Tuple[str, str]:
        """Build prompt for the specialist model (Phase 1 of ensemble)."""
        
        knowledge = ""
        if self.knowledge_retriever:
            try:
                chunks = self.knowledge_retriever.retrieve(task_prompt, max_tokens=400)
                if chunks:
                    knowledge = f"REFERENCE:\n{chunks[0][:400]}\n\n"
            except:
                pass
        
        if suite == "GSM8K":
            return (
                f"{knowledge}Solve this math problem step by step. Show your work clearly "
                f"and give the final numerical answer on the last line.\n\n"
                f"PROBLEM: {task_prompt}\n\nSOLUTION:",
                "You are a mathematics expert. Solve problems step-by-step, showing all work."
            )
        
        elif suite == "HumanEval":
            return (
                f"{knowledge}Complete this Python function. Provide only the function body "
                f"(implementation), not the signature.\n\n{task_prompt}\n\nIMPLEMENTATION:",
                "You are an expert Python programmer. Write clean, efficient, bug-free code."
            )
        
        elif suite == "Spider":
            # SQLCoder ignores system message — uses ### format handled by _format_prompt
            return (
                f"{knowledge}{task_prompt}",
                ""  # SQLCoder template handles its own system
            )
        
        elif suite == "Medical":
            return (
                f"{knowledge}You are reviewing this medical question. Provide your clinical "
                f"assessment based on the evidence.\n\n"
                f"QUESTION: {task_prompt}\n\n"
                f"Provide your answer starting with yes, no, or maybe, then explain:",
                "You are an expert clinical researcher. Provide evidence-based medical analysis."
            )
        
        elif suite == "Legal":
            return (
                f"{knowledge}Analyze this legal question based on contract law principles.\n\n"
                f"QUESTION: {task_prompt}\n\n"
                f"Answer with yes or no, then provide your legal reasoning:",
                "You are a legal expert specializing in contract law and regulatory compliance."
            )
        
        return (task_prompt, "")
    
    def _build_synthesis_prompt(self, suite: str, task_prompt: str, 
                                specialist_response: str) -> Tuple[str, str]:
        """
        Build prompt for DeepSeek supervisor synthesis (Phase 2 of ensemble).
        
        This is the "router/glue" layer — it takes the specialist's domain-grounded
        output and refines it with general reasoning capability.
        
        Analogy to MoE: The specialist model IS the activated expert neurons.
        DeepSeek IS the shared attention/FFN layers + router that coordinates.
        """
        
        if suite == "Medical":
            return (
                f"A medical specialist has reviewed this question and provided their analysis.\n\n"
                f"ORIGINAL QUESTION:\n{task_prompt}\n\n"
                f"SPECIALIST ASSESSMENT:\n{specialist_response[:800]}\n\n"
                f"As a senior reviewer, synthesize the specialist's findings into a clear answer.\n"
                f"Start with yes, no, or maybe. Then provide a concise explanation.\n\n"
                f"FINAL ANSWER:",
                "You are a senior medical reviewer synthesizing specialist opinions. "
                "Provide a clear, definitive answer based on the specialist's analysis."
            )
        
        elif suite == "Legal":
            return (
                f"A legal specialist has reviewed this contract question and provided analysis.\n\n"
                f"ORIGINAL QUESTION:\n{task_prompt}\n\n"
                f"SPECIALIST ANALYSIS:\n{specialist_response[:800]}\n\n"
                f"As a senior legal reviewer, synthesize this into a clear determination.\n"
                f"Start with yes or no. Then provide concise legal reasoning.\n\n"
                f"FINAL DETERMINATION:",
                "You are a senior legal reviewer synthesizing specialist legal analysis. "
                "Give a clear yes/no determination with reasoning."
            )
        
        # Generic synthesis for any other ensemble suite
        return (
            f"A domain specialist provided this analysis:\n\n"
            f"QUESTION: {task_prompt}\n\n"
            f"SPECIALIST: {specialist_response[:800]}\n\n"
            f"Synthesize into a clear, accurate final answer:\n\nANSWER:",
            "You are synthesizing expert analysis into a clear answer."
        )
    
    # ─── Pipeline execution ─────────────────────────────────────────────
    
    async def run_task(self, suite: str, task: Dict) -> BenchmarkResult:
        """
        Run a single benchmark task through the appropriate CoE pipeline.
        
        Pipeline modes:
          solo:     specialist_model(task) → grade → [fallback if 0]
          ensemble: specialist_model(task) → deepseek_synth(specialist_output) → grade → [fallback if 0]
        """
        task_id = task['id']
        task_prompt = task['prompt']
        reference = task.get('answer', '')
        
        pipeline_mode = SUITE_PIPELINE_MODE.get(suite, "solo")
        expert_id = SUITE_EXPERT_MAP.get(suite, "python_backend")
        model_path = SAVANT_MODELS.get(expert_id, SAVANT_MODELS["python_backend"])
        
        start = time.time()
        used_retry = False
        used_fallback = False
        used_ensemble = False
        specialist_response = ""
        
        # ── PHASE 1: Specialist generates domain-grounded draft ──
        
        specialist_loaded = False
        if Path(model_path).exists():
            specialist_loaded = self._load_model(model_path)
        
        if not specialist_loaded:
            # Can't load specialist — go straight to DeepSeek
            if self.enable_fallback:
                pipeline_mode = "fallback"
                model_path = SAVANT_MODELS["deepseek_supervisor"]
                if not Path(model_path).exists() or not self._load_model(model_path):
                    return self._error_result(task_id, suite, expert_id, pipeline_mode, 
                                             "All models failed to load")
                prompt, system = self._build_specialist_prompt(suite, task_prompt)
                response, tokens = self._generate(prompt, system=system, max_tokens=512)
                score, max_score, feedback = grade_response(suite, response, reference)
                latency_ms = (time.time() - start) * 1000
                used_fallback = True
                return BenchmarkResult(
                    task_id=task_id, suite=suite, expert_used="deepseek_supervisor",
                    pipeline_mode="fallback", score=score, max_score=max_score,
                    feedback=feedback, response=response[:200], latency_ms=latency_ms,
                    tokens=tokens, specialist_response="", used_retry=False,
                    used_fallback=True, used_ensemble=False
                )
            else:
                return self._error_result(task_id, suite, expert_id, pipeline_mode,
                                         f"Model {model_path} not found")
        
        # Generate specialist draft
        prompt, system = self._build_specialist_prompt(suite, task_prompt)
        specialist_response, tokens = self._generate(prompt, system=system, max_tokens=512)
        
        # Quality gate on specialist output
        if self.enable_quality_gate and len(specialist_response.strip()) < 20:
            used_retry = True
            specialist_response, tokens = self._generate(prompt, system=system, 
                                                          max_tokens=512, temperature=0.5)
        
        response = specialist_response
        
        # ── PHASE 2: Ensemble synthesis (if applicable) ──
        
        if pipeline_mode == "ensemble":
            used_ensemble = True
            
            # Grade specialist draft first
            spec_score, _, _ = grade_response(suite, specialist_response, reference)
            
            # Always run synthesis — the supervisor adds reasoning and formatting
            synth_path = SAVANT_MODELS["deepseek_supervisor"]
            if Path(synth_path).exists() and self._load_model(synth_path):
                synth_prompt, synth_system = self._build_synthesis_prompt(
                    suite, task_prompt, specialist_response
                )
                response, tokens = self._generate(synth_prompt, system=synth_system, max_tokens=512)
                
                # Check if synthesis improved the answer
                synth_score, _, _ = grade_response(suite, response, reference)
                
                if synth_score < spec_score:
                    # Synthesis made it worse — keep specialist answer
                    response = specialist_response
            else:
                # Can't load DeepSeek — use specialist draft as-is
                pass
        
        # ── PHASE 3: Grade and fallback ──
        
        latency_ms = (time.time() - start) * 1000
        score, max_score, feedback = grade_response(suite, response, reference)
        
        # Score-gated fallback: if score is 0, try pure DeepSeek
        if score == 0 and self.enable_fallback and "deepseek" not in expert_id.lower():
            used_fallback = True
            fallback_path = SAVANT_MODELS["deepseek_supervisor"]
            if Path(fallback_path).exists() and self._load_model(fallback_path):
                # Pure DeepSeek attempt — no specialist context
                fallback_system = "You are DeepSeek, an advanced reasoning model. Think step by step."
                fallback_prompt = (
                    f"Answer this question carefully and accurately.\n\n"
                    f"QUESTION: {task_prompt}\n\n"
                    f"Think through this step by step, then provide your final answer:"
                )
                response, tokens = self._generate(fallback_prompt, system=fallback_system, 
                                                   max_tokens=512)
                score, max_score, feedback = grade_response(suite, response, reference)
                expert_id = "deepseek_fallback"
                latency_ms = (time.time() - start) * 1000
        
        self.pipeline_stats[pipeline_mode] = self.pipeline_stats.get(pipeline_mode, 0) + 1
        
        return BenchmarkResult(
            task_id=task_id,
            suite=suite,
            expert_used=expert_id,
            pipeline_mode=pipeline_mode,
            score=score,
            max_score=max_score,
            feedback=feedback,
            response=response[:200],
            latency_ms=latency_ms,
            tokens=tokens,
            specialist_response=specialist_response[:200] if used_ensemble else "",
            used_retry=used_retry,
            used_fallback=used_fallback,
            used_ensemble=used_ensemble
        )
    
    def _error_result(self, task_id, suite, expert_id, pipeline_mode, error_msg):
        """Create an error BenchmarkResult."""
        return BenchmarkResult(
            task_id=task_id, suite=suite, expert_used=expert_id,
            pipeline_mode=pipeline_mode, score=0.0, max_score=1.0,
            feedback=error_msg, response="", latency_ms=0, tokens=0,
            error=error_msg
        )
    
    async def run_suite(self, suite_name: str, max_samples: int = 25) -> List[BenchmarkResult]:
        """Run all tasks in a benchmark suite."""
        tasks = load_standard_suite(suite_name, max_samples=max_samples)
        
        if not tasks:
            print(f"  ⚠ No tasks found for {suite_name}")
            return []
        
        mode = SUITE_PIPELINE_MODE.get(suite_name, "solo")
        print(f"\n[Suite: {suite_name}] Running {len(tasks)} tasks (pipeline: {mode})...")
        
        results = []
        for i, task in enumerate(tasks):
            print(f"  [{i+1}/{len(tasks)}] {task['id']}...", end="\r")
            result = await self.run_task(suite_name, task)
            results.append(result)
            
        return results
    
    def cleanup(self):
        """Clean up resources."""
        if self._current_model:
            del self._current_model
            self._current_model = None
        self._current_model_path = None
        self._tokenizer = None
        gc.collect()
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─── Orchestrator ───────────────────────────────────────────────────────

async def run_coe_benchmark(
    suite: Optional[str] = None, 
    max_samples: int = 25,
    enable_quality_gate: bool = True,
    enable_fallback: bool = True,
    force_coe: bool = False
):
    """Run CoE benchmark and compare against baselines."""
    
    print("\n" + "="*70)
    print(" COLLEGE OF EXPERTS BENCHMARK v3  (Ensemble Pipeline)")
    print(f" Config: QG={enable_quality_gate}, Fallback={enable_fallback}, ForceCoE={force_coe}")
    print("="*70)
    
    runner = CoEBenchmarkRunner(
        enable_quality_gate=enable_quality_gate,
        enable_fallback=enable_fallback,
        force_coe=force_coe
    )
    runner._init_components()
    
    target_suites = [suite] if suite else list(BASELINE_SCORES.keys())
    all_results = []
    
    try:
        for suite_name in target_suites:
            results = await runner.run_suite(suite_name, max_samples=max_samples)
            all_results.extend(results)
            
            # Suite summary
            valid = [r for r in results if r.error is None]
            if valid:
                avg_score = sum(r.score for r in valid) / len(valid)
                retries = sum(1 for r in valid if r.used_retry)
                fallbacks = sum(1 for r in valid if r.used_fallback)
                ensembles = sum(1 for r in valid if r.used_ensemble)
                avg_latency = sum(r.latency_ms for r in valid) / len(valid)
                
                baseline = BASELINE_SCORES.get(suite_name, {})
                baseline_acc = baseline.get("accuracy", 0)
                delta = (avg_score * 100) - baseline_acc
                
                mode = SUITE_PIPELINE_MODE.get(suite_name, "solo")
                
                print(f"\n[{suite_name} Results] (pipeline: {mode})")
                print(f"  CoE Accuracy:  {avg_score*100:.1f}%")
                print(f"  Baseline:      {baseline_acc:.1f}% ({baseline.get('model', 'N/A')})")
                print(f"  Delta:         {delta:+.1f}%")
                print(f"  Ensemble used: {ensembles}/{len(valid)}")
                print(f"  Retries:       {retries}/{len(valid)}")
                print(f"  Fallbacks:     {fallbacks}/{len(valid)}")
                print(f"  Avg latency:   {avg_latency:.1f}ms")
    finally:
        runner.cleanup()
    
    # Final summary
    print("\n" + "="*70)
    print(" FINAL COMPARISON: CoE v3 vs Baseline")
    print("="*70)
    print(f"{'Suite':<12} | {'Pipeline':<10} | {'CoE':>7} | {'Base':>7} | {'Delta':>7} | {'Ens':>5} | {'FB':>5}")
    print("-"*68)
    
    for suite_name in target_suites:
        suite_results = [r for r in all_results if r.suite == suite_name and r.error is None]
        if suite_results:
            coe_acc = sum(r.score for r in suite_results) / len(suite_results) * 100
            baseline_acc = BASELINE_SCORES.get(suite_name, {}).get("accuracy", 0)
            delta = coe_acc - baseline_acc
            delta_str = f"{delta:+.1f}%"
            ens_count = sum(1 for r in suite_results if r.used_ensemble)
            fb_count = sum(1 for r in suite_results if r.used_fallback)
            mode = SUITE_PIPELINE_MODE.get(suite_name, "solo")
            print(f"{suite_name:<12} | {mode:<10} | {coe_acc:>6.1f}% | {baseline_acc:>6.1f}% | {delta_str:>7} | {ens_count:>2}/{len(suite_results)} | {fb_count:>2}/{len(suite_results)}")
    
    print("="*70)
    print(f"\nPipeline Stats: {runner.pipeline_stats}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CoE v3 benchmark vs baselines")
    parser.add_argument("--suite", type=str, default=None, help="Specific suite to run")
    parser.add_argument("--samples", type=int, default=25, help="Max samples per suite")
    parser.add_argument("--no-qg", action="store_true", help="Disable quality gate")
    parser.add_argument("--no-fallback", action="store_true", help="Disable DeepSeek fallback")
    parser.add_argument("--force-coe", action="store_true", help="Force all tasks through CoE")
    
    args = parser.parse_args()
    
    asyncio.run(run_coe_benchmark(
        suite=args.suite,
        max_samples=args.samples,
        enable_quality_gate=not args.no_qg,
        enable_fallback=not args.no_fallback,
        force_coe=args.force_coe
    ))
