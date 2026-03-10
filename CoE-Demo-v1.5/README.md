# College of Experts - Demo v1.5 (Ollama + ONNX)

This repository contains the standalone, hardware-agnostic demonstration (v1.5) of the College of Experts AI framework. It leverages **Ollama** for hosting large Mixture-of-Experts (MoE) specialist models alongside an **ONNX-based** local Supervisor model.

This framework is built for accessibility, running efficiently on consumer hardware (Windows Copilot+ PCs, AMD APUs, Mac M-series, and Nvidia RTX) without complex CUDA dependency hell.

> **Read the Whitepaper:** The theoretical backing for this architecture is available in the included `WHITEPAPER-QWEN3coderPythonWebSpecialistsModelsShowSeperatabilityofIntelligence.md`.

## Prerequisites
1. Install [Ollama](https://ollama.com/) on your system.
2. Ensure you have Python 3.10+ installed.

## 1. Prepare the Specialist Models
This system routes requests to domain-specific LLMs (Specialists). You must pull these models into Ollama before running the demo.

Open your terminal and run:
*(Note: Replace with exact HuggingFace run commands when models are published)*
```bash
ollama run hf.co/JThomas-CoE/CoE-python2-40b-A3b:q4_K_M
ollama run hf.co/JThomas-CoE/CoE-WEB2-40b-A3b:q4_K_M
```
*(You may `Ctrl+C` immediately once the downloads finish; Ollama will keep them available in its library).*

## 2. Install the Base Environment
Clone this repository and install the hardware-agnostic core requirements:

```bash
git clone https://github.com/JThomas-CoE/College-of-Experts-AI.git
cd "College-of-Experts-AI"
pip install -r requirements.txt
```

## 3. Install your Hardware specific ONNX Provider (CRITICAL)
The **Supervisor Model** (which routes tasks) runs natively in Python using the ONNX Runtime for blazing-fast inference that does not compete with Ollama for VRAM. You *must* install the specific ONNX execution provider for your hardware.

**For AMD APUs / Windows Copilot+ PCs (DirectML):**
```bash
pip install onnxruntime-directml
```

**For Nvidia RTX GPUs (CUDA):**
```bash
pip install onnxruntime-gpu
```

**For Mac / CPU-Only Fallback:**
```bash
pip install onnxruntime
```

## 4. Run the Framework
Start the interactive College of Experts terminal experience:

```bash
python main.py
```

> **Note on First Run:** The very first time you execute `main.py`, the system will download the **BAAI/bge-m3** embedding model (via sentence-transformers) and compile vectorized embeddings for both the **output templates** and the **specialist skills** libraries. This initial startup may take a minute or two depending on your hardware. Subsequent launches load instantly from cache.
>
> During a session you will occasionally see `[TEMPLATE]` and `[SKILL]` log lines — these indicate the framework matched your query against a known output pattern or reasoning guidance entry. This is the context-enrichment layer in action.

Type `/help` for a list of shell commands inside the CoE terminal. Enjoy exploring the separability of machine intelligence!

---

## 5. Customising the Framework

### Output Templates (`config/framework_templates/`)

Templates constrain the **shape of the specialist's output** — they are injected into the user prompt as structural scaffolds.  The library lives in `config/framework_templates/all_templates.json`.

Adding a new template: append an entry to `all_templates.json` and restart (the embedding cache is rebuilt automatically):

```json
{
  "id": "my_template",
  "domain": "code",
  "tags": ["python", "cli"],
  "title": "Python CLI Script",
  "description": "Command-line tool, argparse, entry point, main guard",
  "strength": "strong",
  "scaffold_text": "Structure your output as:\n1. Imports\n2. Argument parser\n3. Main function\n4. `if __name__ == '__main__':` guard"
}
```

Fields: `id` (unique snake_case), `domain` (`code`/`web`), `tags` (retrieval hints), `title`, `description` (embedded for semantic search), `strength` (`strong`/`loose`), `scaffold_text` (injected verbatim).

---

### Specialist Skills (`config/framework_skills/`)

Skills inject **advisory reasoning guidance** into the specialist's *system* prompt — they influence *how* the specialist thinks, not what the output looks like.  The library lives in `config/framework_skills/all_skills.json`.  See [`config/framework_skills/README.md`](config/framework_skills/README.md) for the full schema reference and authoring guide.

---

## 6. Known Limitations

### Grader hallucination

The quality-evaluation step uses the specialist model itself (at near-zero temperature) to grade its own output.  This is a **known failure mode**: the same model that generated a response can confabulate a FAIL reason that does not match the actual content — for example, claiming a decorator is absent when it is present with arguments, or citing a missing optional optimisation as a correctness failure.

The deterministic pre-checks (`def` presence, `ast.parse`, HTML structure) catch structurally broken outputs reliably.  The LLM grader layer is best understood as a heuristic signal, not a precise correctness oracle.

**Improved grading is planned for a future release**, including separation of the grader from the generating model and execution-based validation for code tasks.
