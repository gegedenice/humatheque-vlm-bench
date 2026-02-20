# First Session Prompt

Paste this into Claude Code when starting work in this repo:

---

Starting work on ocr-bench. Read CLAUDE.md for full context. The short version:

This is an OCR evaluation toolkit — VLM-as-judge that produces per-dataset leaderboards on Hub. The key insight: "there is no best OCR model, document type determines the winner."

All existing code lives in `~/Documents/projects/uv-scripts/ocr/`. The important files are:
- `ocr-vllm-judge.py` — offline VLM judge with jury mode (the core engine)
- `ocr-jury-bench.py` — API judge via HF Inference Providers
- `ocr-human-eval.py` — blind A/B human validation Gradio app
- `ocr-bench-run.py` — orchestrator for parallel OCR jobs via HF Jobs
- `OCR-BENCHMARK.md` — 31KB design doc with methodology and all results (becomes README)

First things first:
1. Read `OCR-BENCHMARK.md` and `SHIPPING.md` in `~/Documents/projects/uv-scripts/ocr/` for full context
2. Read the existing judge scripts to understand the current implementation
3. Quick investigation: look at Inspect AI (https://inspect.aisi.org.uk/) — could our VLM judge work as an Inspect scorer? Specifically check if jury mode, ELO rankings, and Hub publishing can work within their framework. This is a go/no-go decision before we commit to package structure.
4. Based on the Inspect decision, init the uv project and start porting the eval scripts

The OCR model scripts (glm-ocr.py, deepseek-ocr-vllm.py, etc.) stay on Hub in uv-scripts/ocr. This repo is eval tooling only.
