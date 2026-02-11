import os
import requests
import pandas as pd
from pathlib import Path

# List of 31 tasks from LegalBench
TASKS = [
    "abercrombie",
    "canada_tax_court_outcomes",
    "citation_prediction_classification",
    "consumer_contracts_qa",
    "contract_nli_confidentiality_of_agreement",
    "contract_nli_explicit_identification",
    "contract_nli_limited_use",
    "contract_nli_no_licensing",
    "contract_nli_notice_on_compelled_disclosure",
    "contract_nli_permissible_acquirement_of_similar_information",
    "contract_nli_permissible_copy",
    "contract_nli_permissible_development_of_similar_information",
    "contract_nli_permissible_post-agreement_possession",
    "contract_nli_return_of_confidential_information",
    "contract_nli_sharing_with_employees",
    "contract_nli_sharing_with_third-parties",
    "contract_nli_survival_of_obligations",
    "contract_qa",
    "corporate_lobbying",
    "cuad_affiliate_license-licensee",
    "cuad_anti-assignment",
    "cuad_audit_rights",
    "cuad_cap_on_liability",
    "cuad_change_of_control",
    "cuad_effective_date",
    "cuad_exclusivity",
    "cuad_expiration_date",
    "cuad_governing_law",
    "cuad_insurance",
    "cuad_license_grant",
    "hearsay"
]

BASE_URL = "https://huggingface.co/datasets/nguha/legalbench/resolve/main/data"
OUTPUT_DIR = Path("benchmarks/data/legalbench")

def download_tasks():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for task in TASKS:
        print(f"Processing {task}...")
        task_dir = OUTPUT_DIR / task
        task_dir.mkdir(exist_ok=True)
        
        url = f"{BASE_URL}/{task}/test.tsv"
        dest = task_dir / "test.tsv"
        
        if dest.exists():
            print(f"  Already exists at {dest}")
            continue
            
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(dest, "wb") as f:
                    f.write(response.content)
                print(f"  Downloaded to {dest}")
            else:
                print(f"  Failed to download {task}: Status {response.status_code}")
        except Exception as e:
            print(f"  Error downloading {task}: {e}")

if __name__ == "__main__":
    download_tasks()
