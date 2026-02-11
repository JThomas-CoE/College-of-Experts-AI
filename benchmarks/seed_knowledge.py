"""
Seed Knowledge - Populates ChromaDB with grounded knowledge for experts.
Used to demonstrate RAG improvements in Phase 2.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.vector_backbone import VectorBackbone

def seed_legal_knowledge(vb: VectorBackbone):
    print("Seeding Legal Knowledge...")
    documents = [
        "Hearsay is an out-of-court statement offered to prove the truth of the matter asserted. Exceptions include excited utterances and business records.",
        "The Abercrombie classification for trademarks includes: Generic, Descriptive, Suggestive, Arbitrary, and Fanciful. Generic marks cannot be trademarked.",
        "A bilateral contract is an agreement where both parties exchange mutual promises. A unilateral contract involves a promise in exchange for performance.",
        "Under the Four Corners rule, court's interpretation of a document is limited to the document's own terms, without external evidence.",
        "Negligence requires four elements: Duty of care, Breach of that duty, Causation (factual and proximate), and Damages."
    ]
    metadatas = [{"source": "LegalBench_Core"} for _ in range(len(documents))]
    ids = [f"legal_core_{i}" for i in range(len(documents))]
    
    vb.ingest_documents("legal", documents, metadatas, ids)
    print(f"  Done. Legal collection now has {vb.get_stats().get('legal', 0)} entries.")

def seed_medical_knowledge(vb: VectorBackbone):
    print("Seeding Medical Knowledge...")
    documents = [
        "Type 1 Diabetes is an autoimmune destruction of beta cells in the pancreas, leading to absolute insulin deficiency.",
        "Type 2 Diabetes involves peripheral insulin resistance and inadequate compensatory insulin secretion.",
        "Myocardial Infarction occurs when blood flow to the heart muscle is blocked, leading to ischemia and tissue necrosis.",
        "Penicillin works by inhibiting bacterial cell wall synthesis through binding to penicillin-binding proteins (PBPs).",
        "The loop of Henle in the kidney generates an osmotic gradient in the medulla to allow for urine concentration."
    ]
    metadatas = [{"source": "PubMedQA_Reference"} for _ in range(len(documents))]
    ids = [f"med_core_{i}" for i in range(len(documents))]
    
    vb.ingest_documents("medical", documents, metadatas, ids)
    print(f"  Done. Medical collection now has {vb.get_stats().get('medical', 0)} entries.")

def seed_sql_knowledge(vb: VectorBackbone):
    print("Seeding SQL Knowledge...")
    documents = [
        "To join two tables, use the JOIN keyword followed by the ON clause to specify the joining columns.",
        "The GROUP BY clause is used with aggregate functions like COUNT, SUM, AVG to group the result-set by one or more columns.",
        "The HAVING clause was added to SQL because the WHERE keyword could not be used with aggregate functions.",
        "A LEFT JOIN returns all records from the left table, and the matched records from the right table. If no match, NULL is returned for the right side."
    ]
    metadatas = [{"source": "SQLCoder_BestPractices"} for _ in range(len(documents))]
    ids = [f"sql_core_{i}" for i in range(len(documents))]
    
    vb.ingest_documents("sql", documents, metadatas, ids)
    print(f"  Done. SQL collection now has {vb.get_stats().get('sql', 0)} entries.")

def seed_python_knowledge(vb: VectorBackbone):
    print("Seeding Python Knowledge...")
    documents = [
        "In modern Python (3.9+), use built-in collection types for type hinting: list[str], dict[str, int]. Import Optional and Union from typing.",
        "List comprehensions are faster and more readable than for-loops for simple mappings: [x**2 for x in data if x > 0].",
        "Use contextlib.contextmanager to create small, reusable context managers for resource handling without boiler plate classes.",
        "The functools.lru_cache decorator provides easy memoization: @lru_cache(maxsize=128) def expensive_func(n): ...",
        "Use f-strings for efficient and readable string formatting: f'Value is {val:.2f}'. Avoid .format() in performance-critical code.",
        "To run model inference with onnxruntime-genai (OGA): Create a GeneratorParams object, set max_length, then loop generator.generate_next_token() until is_done().",
        "Accessing environment variables safely: os.getenv('API_KEY', default_value) ensures the app doesn't crash if the variable is missing.",
        "Use dataclasses for clean, type-checked data objects: @dataclass class User: id: int; name: str."
    ]
    metadatas = [{"source": "Python_Best_Practices"} for _ in range(len(documents))]
    ids = [f"py_core_{i}" for i in range(len(documents))]
    
    vb.ingest_documents("python", documents, metadatas, ids)
    print(f"  Done. Python collection now has {vb.get_stats().get('python', 0)} entries.")

if __name__ == "__main__":
    vb = VectorBackbone("data/vector_db")
    seed_legal_knowledge(vb)
    seed_medical_knowledge(vb)
    seed_sql_knowledge(vb)
    seed_python_knowledge(vb)
    print("\nKnowledge Corpus Seeding Complete.")
