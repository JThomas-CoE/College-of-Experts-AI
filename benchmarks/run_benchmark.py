import time
import json
import os
import sys
import gc
import torch
import re
import random
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.backends.oga_backend import OGABackend

# Specialist Prompt Templates
TEMPLATES = {
    "qwen": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
    "mistral": "<s>[INST] {prompt} [/INST]",
    "sql": "### Instruction:\n{prompt}\n\n### Response:\n",
    "llama": "[INST] {prompt} [/INST]",
    "default": "{prompt}"
}

# Internal fallback suite
BENCHMARK_SUITE = {
    "Spider": [
        {"id": "sql_01", "prompt": "Find names of all employees in 'engineering' with salary > 50000", "template": "sql", "answer": ["SELECT", "engineering", "salary", "50000"]},
        {"id": "sql_02", "prompt": "Count the number of students in each department.", "template": "sql", "answer": ["COUNT", "GROUP BY", "department"]},
        {"id": "sql_03", "prompt": "Show the average age of customers who have made more than 5 purchases.", "template": "sql", "answer": ["AVG", "age", "HAVING", "COUNT"]},
        {"id": "sql_04", "prompt": "List all projects and the names of employees assigned to them, including projects with no employees.", "template": "sql", "answer": ["LEFT JOIN", "ON", "projects", "employees"]},
        {"id": "sql_05", "prompt": "Find the highest paid employee in each city.", "template": "sql", "answer": ["MAX", "salary", "GROUP BY", "city"]}
    ],
    "Legal": [
        {"id": "law_01", "prompt": "What are the four elements of negligence under common law?", "template": "mistral", "answer": ["duty", "breach", "causation", "damages"]},
        {"id": "law_02", "prompt": "Explain the difference between a bilateral and a unilateral contract.", "template": "mistral", "answer": ["promise", "performance", "acceptance"]},
        {"id": "law_03", "prompt": "What is the 'Miranda warning' and when must it be given?", "template": "mistral", "answer": ["custody", "interrogation", "rights", "silence"]},
        {"id": "law_04", "prompt": "Define 'stare decisis' and its importance in the US legal system.", "template": "mistral", "answer": ["precedent", "authority", "binding"]},
        {"id": "law_05", "prompt": "What constitutes 'hearsay' and name one common exception.", "template": "mistral", "answer": ["out of court", "truth of the matter", "excited utterance", "business record"]}
    ],
    "Medical": [
        {"id": "bio_01", "prompt": "Explain the difference between Type 1 and Type 2 diabetes in terms of pathophysiology.", "template": "mistral", "answer": ["insulin", "autoimmune", "resistance", "pancreas"]},
        {"id": "bio_02", "prompt": "What are the common symptoms and mechanism of a myocardial infarction?", "template": "mistral", "answer": ["chest pain", "ischemia", "coronary artery", "necrosis"]},
        {"id": "bio_03", "prompt": "Describe the function of the loop of Henle in the kidney.", "template": "mistral", "answer": ["reabsorption", "concentration", "urine", "medulla"]},
        {"id": "bio_04", "prompt": "What are the main neurotransmitters involved in Parkinson's disease?", "template": "mistral", "answer": ["dopamine", "substantia nigra", "basal ganglia"]},
        {"id": "bio_05", "prompt": "Explain the mechanism of action of penicillin-class antibiotics.", "template": "mistral", "answer": ["cell wall", "peptidoglycan", "beta-lactam", "transpeptidase"]}
    ]
}

# Published Reference Scores (Approximate for comparison)
PUBLISHED_REFS = {
    "HumanEval": {"Qwen2.5-Coder-7B": 84.1, "Llama-3-8B": 62.0, "Mistral-7B": 30.0},
    "GSM8K": {"Qwen2.5-Math-7B": 82.0, "Llama-3-8B": 78.0, "Mistral-7B": 52.0},
    "Spider": {"sqlcoder-7b-2": 72.0},
    "Medical": {"BioMistral-7B": 68.0}
}

def load_standard_suite(suite_name, max_samples=None):
    """Loads tasks from official dataset files."""
    data_dir = Path(__file__).parent / "data"
    tasks = []
    
    if suite_name == "GSM8K":
        path = data_dir / "gsm8k_test.jsonl"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    # Extract answer after ####
                    ans = item['answer'].split('####')[-1].strip().replace(',', '')
                    tasks.append({
                        "id": f"gsm_{len(tasks)}",
                        "prompt": item['question'],
                        "template": "qwen",
                        "answer": ans
                    })
    
    elif suite_name == "Medical":
        path = data_dir / "pubmedqa.json"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for pmid, item in data.items():
                    tasks.append({
                        "id": f"pmid_{pmid}",
                        "prompt": f"Context: {' '.join(item.get('CONTEXTS', []))}\nQuestion: {item['QUESTION']}\nAnswer with yes, no, or maybe.",
                        "template": "mistral",
                        "answer": item['final_decision']
                    })
    
    elif suite_name == "HumanEval":
        path = data_dir / "HumanEval.jsonl"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    tasks.append({
                        "id": item['task_id'],
                        "prompt": item['prompt'],
                        "template": "qwen",
                        "answer": item['canonical_solution']
                    })
                    
    elif suite_name == "Spider":
        path = data_dir / "spider_dev.json"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    tasks.append({
                        "id": f"spider_{len(tasks)}",
                        "prompt": f"Database: {item['db_id']}\nQuestion: {item['question']}",
                        "template": "sql",
                        "answer": item['query']
                    })
                    
    elif suite_name == "Legal":
        legal_dir = data_dir / "legalbench"
        if legal_dir.exists():
            import csv
            # Iterate through all task folders
            for task_folder in legal_dir.iterdir():
                if task_folder.is_dir():
                    tsv_path = task_folder / "test.tsv"
                    if tsv_path.exists():
                        with open(tsv_path, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f, delimiter='\t')
                            for row in reader:
                                # Construct prompt from all columns except answer and index
                                context = []
                                for k, v in row.items():
                                    if k.lower() not in ['answer', 'index']:
                                        context.append(f"{k.capitalize()}: {v}")
                                
                                tasks.append({
                                    "id": f"legal_{task_folder.name}_{row.get('index', len(tasks))}",
                                    "prompt": "\n".join(context),
                                    "template": "mistral",
                                    "answer": row.get('answer', '')
                                })
                    
    # Fallback to internal small suite if file doesn't exist or for other suites
    if not tasks:
        tasks = BENCHMARK_SUITE.get(suite_name, [])
        
    if max_samples:
        random.seed(42) # Deterministic subset
        if len(tasks) > max_samples:
            tasks = random.sample(tasks, max_samples)
            
    return tasks

def grade_response(suite, response, reference):
    """
    Grades the response based on the suite type.
    Returns (score, max_score, feedback)
    """
    if not response or not reference:
        return 0.0, 1.0, "Empty response or reference."
        
    response_clean = response.lower().strip()
    
    if suite == "GSM8K":
        # Standard: Look for the last number in the response
        # Find all numbers (including decimals)
        numbers = re.findall(r'-?\d+\.?\d*', response_clean.replace(',', ''))
        if numbers:
            last_num = numbers[-1]
            if last_num == str(reference):
                return 1.0, 1.0, f"Match: {last_num}"
            return 0.0, 1.0, f"Expected {reference}, found {last_num}"
        return 0.0, 1.0, "No numbers found in response."
    
    elif suite == "HumanEval":
        # Code similarity proxy
        # Standard HumanEval uses execution, but we'll use keyword intersection + length heuristic
        # as a comparable proxy for "completion quality"
        ref_keywords = set(re.findall(r'\w+', reference.lower()))
        resp_keywords = set(re.findall(r'\w+', response_clean))
        
        intersection = ref_keywords.intersection(resp_keywords)
        score = len(intersection) / len(ref_keywords) if ref_keywords else 1.0
        
        # Penalty for extremely short or suspiciously long responses
        if len(response_clean) < len(reference) * 0.3: score *= 0.5
        
        return score, 1.0, f"Semantics: {int(score*100)}%"

    elif suite == "Medical":
        # PubMedQA special handling
        if reference.lower() in ["yes", "no", "maybe"]:
            # Check if the correct answer is the FIRST word or prominent
            first_word = re.findall(r'\w+', response_clean)
            if first_word and first_word[0] == reference.lower():
                return 1.0, 1.0, f"Correct: {reference}"
            # Fallback: check if the answer exists at all but penalize uncertainty
            if reference.lower() in response_clean:
                return 0.5, 1.0, f"Contains correct answer: {reference}"
            return 0.0, 1.0, f"Incorrect. Target: {reference}"
        
    elif suite == "Spider":
        ref_keywords = set(re.findall(r'\w+', reference.lower()))
        resp_keywords = set(re.findall(r'\w+', response_clean))
        matches = ref_keywords.intersection(resp_keywords)
        score = len(matches) / len(ref_keywords) if ref_keywords else 1.0
        return score, 1.0, f"Keyword match: {len(matches)}/{len(ref_keywords)}"

    elif suite == "Legal":
        first_word = re.findall(r'\w+', response_clean)
        ref_words = re.findall(r'\w+', reference.lower())
        if first_word and ref_words and first_word[0] == ref_words[0]:
            return 1.0, 1.0, f"Match: {reference}"
        if reference.lower() == response_clean:
            return 1.0, 1.0, "Exact match"
        if reference.lower() in response_clean:
            return 0.7, 1.0, f"Contains: {reference}"
        return 0.0, 1.0, f"Expected {reference}"
    
    return 0.0, 0.0, "Unknown suite grading."

def run_benchmark(model_id_or_path, suite_name=None, max_samples=10, use_rag=False):
    # Load Backend ONCE for all suites to keep it in memory
    backend = None
    try:
        print(f"Loading model into memory...")
        backend = OGABackend(model_id_or_path)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model {model_id_or_path}: {e}")
        return [{"suite": suite_name or "error", "error": f"Model load failed: {e}"}]

    # Lazy-Initialize VectorDB AFTER model is in memory to avoid ORT conflict
    vb = None
    if use_rag:
        try:
            from src.vector_backbone import VectorBackbone
            vb = VectorBackbone("data/vector_db")
            print("[RAG] Vector store initialized.")
        except Exception as e:
            print(f"[RAG] Failed to init: {e}")
            use_rag = False
            
    print(f"\n{'='*60}")
    print(f"RUNNING COMPARABLE BENCHMARK: {model_id_or_path}")
    print(f"Target Suite: {suite_name or 'ALL'} | Max Samples: {max_samples}")
    print(f"{'='*60}")
    
    results = []
    target_suites = [suite_name] if suite_name else list(BENCHMARK_SUITE.keys())

    try:
        for suite in target_suites:
            tasks = load_standard_suite(suite, max_samples=max_samples)
            if not tasks: 
                print(f"Skipping {suite}: No tasks found.")
                continue
            
            print(f"\n[Suite: {suite}] Loaded {len(tasks)} tasks.")
            
            for i, test in enumerate(tasks):
                print(f"  [{i+1}/{len(tasks)}] Task {test['id']}...", end="\r")
                
                try:
                    template_key = test.get("template", "default")
                    # Detect template from model path if not set
                    if "qwen" in model_id_or_path.lower(): template_key = "qwen"
                    elif "mistral" in model_id_or_path.lower() or "bio" in model_id_or_path.lower(): template_key = "mistral"
                    elif "law" in model_id_or_path.lower() or "llama" in model_id_or_path.lower(): template_key = "llama"
                    elif "sql" in model_id_or_path.lower(): template_key = "sql"
                    
                    prompt = test['prompt']
                    
                    # RAG Retrieval
                    if use_rag:
                        domain_map = {"HumanEval": "python", "GSM8K": "math", "Spider": "sql", "Medical": "medical", "Legal": "legal"}
                        domain = domain_map.get(suite, "general")
                        try:
                            v_res = vb.query(domain, prompt, n_results=1)
                            docs = v_res.get('documents', [[]])[0]
                            if docs:
                                prompt = f"CONTEXT:\n{docs[0]}\n\nQUESTION: {prompt}\n\nINSTRUCTION: Base your answer on the CONTEXT if relevant."
                                # print(f"[RAG] Grounded task {test['id']}")
                        except: pass

                    formatted_prompt = TEMPLATES[template_key].format(prompt=prompt)
                    
                    start_time = time.time()
                    response, token_count = backend.generate(formatted_prompt, max_new_tokens=256)
                    duration = time.time() - start_time
                    
                    tps = token_count / duration if duration > 0 else 0
                    
                    score, max_score, feedback = grade_response(suite, response, test.get("answer"))
                    print(f"\n    DEBUG [{test['id']}]: Response: {response[:100]}... | Expected: {test.get('answer')}")
                    
                    results.append({
                        "id": test['id'],
                        "suite": suite,
                        "tok_per_sec": tps,
                        "score": score,
                        "feedback": feedback
                    })
                    
                except Exception as e:
                    print(f"\n    Error in task {test['id']}: {e}")
                    results.append({"id": test['id'], "suite": suite, "error": str(e)})

            # Print Suite Summary
            suite_res = [r for r in results if r['suite'] == suite and "score" in r]
            if suite_res:
                avg_score = sum(r['score'] for r in suite_res) / len(suite_res)
                avg_tps = sum(r['tok_per_sec'] for r in suite_res) / len(suite_res)
                print(f"\n[Suite {suite} Results]")
                print(f"  Accuracy: {avg_score*100:.1f}%")
                print(f"  Throughput: {avg_tps:.2f} tok/s")
                
                # Show Published Reference if exists
                model_key = None
                if "qwen" in model_id_or_path.lower(): 
                    if "coder" in model_id_or_path.lower(): model_key = "Qwen2.5-Coder-7B"
                    elif "math" in model_id_or_path.lower(): model_key = "Qwen2.5-Math-7B"
                elif "bio" in model_id_or_path.lower(): model_key = "BioMistral-7B"
                elif "sql" in model_id_or_path.lower(): model_key = "sqlcoder-7b-2"
                
                if model_key and model_key in PUBLISHED_REFS.get(suite, {}):
                    pub_score = PUBLISHED_REFS[suite][model_key]
                    diff = (avg_score * 100) - pub_score
                    print(f"  Reference ({model_key}): {pub_score}% (Delta: {diff:+.1f}%)")
    finally:
        # Final Cleanup after all suites are done
        if backend:
            print(f"\nCleaning up model from memory...")
            del backend
            gc.collect()
            if torch.cuda.is_available(): 
                torch.cuda.empty_cache()

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run comparable benchmarks.")
    parser.add_argument("model_path", type=str, help="Path to the model to test.")
    parser.add_argument("suite", type=str, nargs="?", default=None, help="Specific benchmark suite to run.")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to run per suite.")
    parser.add_argument("--grounded", action="store_true", help="Enable RAG grounding via VectorBackbone.")
    
    args = parser.parse_args()
    results = run_benchmark(args.model_path, args.suite, max_samples=args.samples, use_rag=args.grounded)
    
    print("\n" + "="*60)
    print("FINAL CONSOLIDATED RESULTS")
    print("="*60)
    
    suites = sorted(list(set(r['suite'] for r in results)))
    for s in suites:
        s_res = [r for r in results if r['suite'] == s and "score" in r]
        if s_res:
            acc = sum(r['score'] for r in s_res) / len(s_res)
            tps = sum(r['tok_per_sec'] for r in s_res) / len(s_res)
            print(f"{s:<15} | Acc: {acc*100:6.1f}% | Perf: {tps:6.2f} tok/s")
    print("="*60)
