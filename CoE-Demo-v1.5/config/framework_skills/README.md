# Specialist Skills Library

This directory contains the advisory reasoning guidance library used by the College of Experts framework's skill-injection layer.

---

## What skills do

Skills inject **"how to think" guidance** into the specialist model's *system prompt* before it generates a response.

This is architecturally distinct from output templates:

| Layer | File | Injected into | Effect |
|-------|------|---------------|--------|
| **Skills** | `all_skills.json` | Specialist *system* prompt | Guides reasoning approach |
| **Templates** | `../framework_templates/all_templates.json` | User *task* prompt | Constrains output shape |

When the framework matches a skill, you will see a dim log line like:
```
[SKILL] Python Error Handling (0.62)
```
The number in parentheses is the cosine similarity score.  Retrieval threshold is **0.50** — intentionally loose because guidance is advisory.

---

## Skill schema

Each entry in `all_skills.json` is a JSON object with 6 fields:

```json
{
  "id":          "code_python_error_handling",
  "domain":      "code",
  "tags":        ["python", "errors", "exceptions"],
  "title":       "Python Error Handling",
  "description": "Python exception handling, try/except blocks, specific exception types, context managers",
  "guidance":    "- Catch specific exception types...\n- Use context managers (`with`) for resource cleanup...\n- ..."
}
```

| Field | Type | Purpose |
|-------|------|---------|
| `id` | string | Unique snake_case identifier.  Never reuse or rename an existing ID. |
| `domain` | string | `"code"` or `"web"`.  Skills matching the active specialist's domain receive a **+0.05 cosine boost** during retrieval. |
| `tags` | string[] | Keyword hints for authoring clarity — not directly used in retrieval, but useful for human navigation. |
| `title` | string | Short noun phrase.  Combined with `description` to form the retrieval text: `"{title}. {description}"`. |
| `description` | string | One sentence describing the problem area.  **This is the text that gets embedded** — write it to match how a user would phrase a query, not how you'd categorise the topic. |
| `guidance` | string | Markdown bullet list injected verbatim into the specialist system prompt.  Preceded by a heading: `"Advisory guidance for this task:"`. |

---

## Writing good guidance

The `guidance` field is what the specialist actually reads.  A few principles:

- **Be concrete, not categorical.**  "Use `collections.deque` for O(1) append/popleft" beats "choose the right data structure."
- **Use advisory language.**  "Prefer", "consider", "avoid" — not "must" or "always".  The specialist knows its job; guidance should nudge, not override.
- **5–7 bullets is the sweet spot.**  Too few has no impact; too many dilute attention.
- **Make it specific to the domain.**  A generic "write clean code" bullet adds no value.

---

## Adding a new skill

1. Open `all_skills.json`.
2. Append a new entry to the top-level array.
3. Restart the framework — the embedding cache is rebuilt automatically when the file changes.

Example — adding a Python generators skill:

```json
{
  "id": "code_python_generators",
  "domain": "code",
  "tags": ["python", "generators", "yield", "itertools", "lazy evaluation", "memory efficiency"],
  "title": "Python Generators and Lazy Evaluation",
  "description": "Python generator functions, yield, yield from, generator expressions, itertools, lazy pipelines, memory-efficient iteration",
  "guidance": "- Use generator functions (`yield`) instead of building a full list when only one element is needed at a time\n- Prefer generator expressions `(x for x in ...)` over list comprehensions when the result is immediately consumed\n- Use `yield from` to delegate to a sub-generator cleanly without a manual loop\n- `itertools.chain`, `islice`, `takewhile`, `groupby` compose lazy pipelines without intermediate allocations\n- Send values into a generator with `.send(val)` for coroutine-style state machines\n- Generators are single-pass: if you need to iterate twice, materialise with `list()` or use `itertools.tee` with care"
}
```

---

## Domain coverage (current library)

28 skills across 2 domains:

### Code (16)
| ID | Title |
|----|-------|
| `code_python_error_handling` | Python Error Handling |
| `code_python_type_hints` | Python Type Hints and Docstrings |
| `code_python_algorithm_complexity` | Algorithm Complexity and Performance |
| `code_python_data_structures` | Python Data Structure Selection |
| `code_python_testing` | Python Testing Approach |
| `code_python_security` | Python Security Practices |
| `code_python_concurrency` | Python Concurrency Patterns |
| `code_python_oop_design` | Python OOP and Class Design |
| `code_python_api_design` | Python API and Interface Design |
| `code_python_dependency_management` | Python Dependency Management |
| `code_refactoring_principles` | Refactoring and Code Quality |
| `code_debugging_approach` | Systematic Debugging Approach |
| `code_documentation_standards` | Code Documentation Standards |
| `code_python_numba_jit` | Numba JIT Acceleration |
| `code_python_threadsafe_multiprocessing` | Thread-Safe Multiprocessing |
| `code_python_numpy_vectorization` | NumPy Vectorized Array Operations |

### Web (12)
| ID | Title |
|----|-------|
| `web_semantic_html_accessibility` | Semantic HTML and Accessibility |
| `web_css_layout` | CSS Layout Best Practices |
| `web_javascript_async` | JavaScript Async Patterns |
| `web_performance` | Web Performance Optimization |
| `web_security` | Web Security Fundamentals |
| `web_responsive_design` | Responsive Design Approach |
| `web_react_patterns` | React Component Patterns |
| `web_typescript_usage` | TypeScript Usage Patterns |
| `web_state_management` | Web State Management Strategy |
| `web_api_consumption` | REST API Consumption Patterns |
| `web_css_naming_organisation` | CSS Naming and Organisation |
| `web_node_backend_patterns` | Node.js Backend Patterns |

---

## Extending to new domains

> **Important:** Skills enrich what a specialist *already does well* — they do **not** extend language or domain coverage.  The code specialist is fine-tuned for Python; injecting Go or Rust guidance into it would provide noise, not value.  To cover a genuinely new language or domain you need a new specialist model, not just new skills.

Adding skills for an already-supported specialist domain (e.g. adding more Python skills to `domain: "code"`) requires only editing `all_skills.json` and restarting.

Adding an entirely **new** domain (e.g. `sql`, `devops`) requires both:

1. Skill entries with the new `domain` value in `all_skills.json`.
2. A new specialist model registered in `coe_demo/models.py` for that domain.

The skill injection in `pipeline.py` automatically covers any domain that is not `"supervisor"` — no pipeline changes are needed.

---

## Cache

The embedding vectors are stored in `embedding_cache/skill_vectors.npy` (and a companion metadata file).  This directory is `.gitignore`d — it is machine-specific and rebuilt automatically.  Delete it to force a full rebuild.
