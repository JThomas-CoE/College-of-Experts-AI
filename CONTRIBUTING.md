# Contributing to College of Experts

Thank you for your interest in contributing! This project is in early development, and we welcome contributions of all kinds.

## Ways to Contribute

### 1. Expert Model Recommendations

Have you found a specialized model that works well for a particular domain? Open an issue or PR to add it to `docs/experts.md`.

**What we're looking for:**
- Models with clear domain specialization
- Available in quantized formats (GGUF preferred)
- Sizes from 1B-15B parameters (loadable on consumer hardware)
- Open weights with permissive licenses

### 2. Memory Layer Implementations

The shared memory backbone is a key component. We welcome:
- Alternative storage backends (Redis, DuckDB, etc.)
- Vector search integrations
- Memory compression strategies
- Persistence optimizations

### 3. Router Improvements

The router is critical for user experience:
- Better scope classification
- Faster expert selection algorithms
- Improved transition detection
- Multi-lingual support

### 4. Documentation

- Tutorials and guides
- Architecture explanations
- Benchmark results
- Video walkthroughs

## Development Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/college-of-experts.git
cd college-of-experts

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest
```

## Code Style

- Python code follows PEP 8
- Use type hints for function signatures
- Include docstrings for public functions
- Format with `black` and `isort`

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Run the test suite
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request

## Issue Guidelines

When opening an issue:
- Use a clear, descriptive title
- Include your environment (OS, Python version, GPU)
- For bugs: steps to reproduce, expected vs actual behavior
- For features: use case and proposed implementation

## Questions?

Open a Discussion for general questions about the architecture or implementation approach.

---

*We're building something new here. Your contributions help shape what local AI can become.*
