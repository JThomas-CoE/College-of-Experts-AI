import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.run_benchmark import run_benchmark

def run_baseline_tests():
    experts_to_test = {
        "HumanEval": "models/Qwen2.5-Coder-7B-DML",
        "GSM8K": "models/Qwen2.5-Math-7B-DML",
        "Medical": "models/BioMistral-7B-DML",
        "Spider": "models/sqlcoder-7b-2-DML",
        "Legal": "models/law-LLM-DML"
    }
    
    overall_results = {}
    
    for suite, model_path in experts_to_test.items():
        results = run_benchmark(model_path, suite, max_samples=25)
        overall_results[suite] = results
        
    print("\n" + "="*60)
    print("BASELINE PERFORMANCE OVERVIEW")
    print("="*60)
    for suite, items in overall_results.items():
        valid_res = [i for i in items if "score" in i]
        if valid_res:
            acc = sum(i['score'] for i in valid_res) / len(valid_res)
            tps = sum(i['tok_per_sec'] for i in valid_res) / len(valid_res)
            print(f"{suite:<15} | Accuracy: {acc*100:6.1f}% | Speed: {tps:6.2f} tok/s")
    print("="*60)

if __name__ == "__main__":
    run_baseline_tests()
