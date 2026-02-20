# CLAUDE.md — ocr-bench

OCR model evaluation toolkit. VLM-as-judge with per-dataset leaderboards on Hugging Face Hub.

## What This Project Does

Lets anyone answer: **"Which OCR model works best for MY documents?"**

OCR model rankings change depending on document type — manuscript cards, printed books, historical texts, tables all produce different winners. A single leaderboard is misleading. This tool lets you create a leaderboard for your specific collection.

## Desired Outcomes

1. **Evaluate OCR models on any dataset** — run models, then judge output quality using VLM-as-judge. Produces ELO rankings and per-dimension scores.
2. **Customizable evaluation criteria** — presets for common document types (index cards, manuscripts, tables, printed books), custom prompts for power users, configurable ignore lists.
3. **Per-dataset leaderboards on Hub** — every evaluation run publishes a leaderboard dataset to Hub. The judge prompt and metadata are stored for reproducibility.
4. **Leaderboard viewer on Hub** — reads any evaluation dataset matching the standard schema. Approach TBD.
5. **Blog post** — "There Is No Best OCR Model". Ties into FineCorpus narrative.

## Tooling

- **uv** for project management and running scripts
- **ruff** for linting and formatting

## Repo Architecture

### Now: Ship fast

GitHub repo with the eval tooling as a uv project. OCR model scripts stay on Hub in `uv-scripts/ocr/` where they already work with `hf jobs uv run`. No sync machinery — just cross-reference.

```
ocr-bench/                        ← GitHub repo
├── README.md                     ← polished from OCR-BENCHMARK.md
├── CLAUDE.md
├── pyproject.toml                ← uv project
└── src/ or ocr_bench/            ← eval tooling (judge, publishing, presets)
```

OCR model scripts stay at `davanstrien/uv-scripts` on Hub (14 scripts, all reviewed).

### Later (if the project gets traction): Consolidate

Move OCR model scripts into this repo under a `uv-scripts/` subdirectory. Use [hub-sync](https://github.com/huggingface/hub-sync) GitHub Action to sync that subdirectory to Hub so `hf jobs uv run` keeps working. One source of truth, automatic sync. Only worth the effort if people are actually using the benchmark.

## What Already Exists

All current code is in `~/Documents/projects/uv-scripts/ocr/`. Key files to port:

| Script | What it does |
|--------|-------------|
| `ocr-vllm-judge.py` | Offline VLM judge. Jury mode (multiple judges vote). Structured output via xgrammar — 0 parse failures over 300+ comparisons. ELO. |
| `ocr-jury-bench.py` | API-based judge via HF Inference Providers. Default: Kimi K2.5 via Novita. |
| `ocr-human-eval.py` | Blind A/B Gradio app for human validation of judge quality. |
| `ocr-bench-run.py` | Orchestrator: launches N OCR models via HF Jobs, auto-merges PRs. |
| `OCR-BENCHMARK.md` | 31KB design doc — methodology, results, findings. Becomes README. |
| `SHIPPING.md` | Architecture decisions. |
| `REVIEW.md` | Review notes for all 14 OCR model scripts. |
| 14 OCR model scripts | All reviewed + fixed 2026-02-16. Core 4: DeepSeek-OCR, GLM-OCR, LightOnOCR-2, DoTS.ocr. |

### Results on Hub
- `davanstrien/ocr-bench-rubenstein-judge` — 50 samples, 300 comparisons, jury of 2
- `davanstrien/ocr-bench-ufo-judge-30b` — cross-validation on UFO-ColPali
- `davanstrien/ocr-bench-rubenstein-judge-kimi-k25` — Kimi K2.5 170B
- Human validation: 30 blind comparisons, Kimi K2.5 matches human rankings best

## Validated Findings (for README/blog)

1. **No single best OCR model** — rankings shuffle by document type
2. **DeepSeek-OCR most consistent** — #1 or #2 across all datasets
3. **Document type > model size** — 0.9B beats 4B on some collections
4. **Judge model size matters** — small judges biased toward verbose output; 170B closest to human
5. **Jury mode works** — eliminates single-judge bias
6. **Structured output reliable** — 0 parse failures at scale via xgrammar

## Prior Art

### DoTS.ocr
`https://github.com/rednote-hilab/dots.ocr/blob/master/tools/elo_score_prompt.py`
Pairwise VLM judge (Gemini), fixed criteria, generous ties, ignores formatting/layout. No jury, no human validation, no customization. Their explicit ignore list is worth adopting as a configurable option.

### Inspect AI (UK AISI)
`https://inspect.aisi.org.uk/`
Open-source eval framework. **Decision (2026-02-20): NO-GO.** Inspect evaluates one model at a time (input → model → score against target). Our pipeline is fundamentally different: pre-computed outputs from N models, pairwise VLM judge, ELO ranking. Adopting Inspect would mean no-op solvers, reimplementing jury logic inside a custom scorer, and computing ELO outside the framework. Not worth the impedance mismatch. Build standalone.

### HF Eval Results System
`https://huggingface.co/docs/hub/en/eval-results`
Decentralized eval results on Hub: benchmark datasets define `eval.yaml`, model repos store scores in `.eval_results/*.yaml`, results appear on model cards + benchmark leaderboards. Useful as a **publishing/discovery layer** — after our judge produces rankings, publish into this system. Not a replacement for our eval tooling. The `evaluation_framework` enum would need an `ocr-bench` entry added.

## Sprint Plan

### Phase 1: Foundation
- [x] Investigate Inspect AI — **NO-GO** (2026-02-20). Build standalone.
- [ ] Init uv project with `src/ocr_bench/` layout
- [ ] Port `ocr-vllm-judge.py` → `src/ocr_bench/judge.py` (core pairwise judge, structured output, ELO)
- [ ] Port `ocr-jury-bench.py` → `src/ocr_bench/jury.py` (API-based judge, jury mode)
- [ ] Port ELO computation + results aggregation as shared module
- [ ] CLI entrypoint: `ocr-bench judge` / `ocr-bench jury`
- [ ] Write README — key finding ("no single best model") as the headline
- [ ] Adapt `OCR-BENCHMARK.md` methodology section for README

### Phase 2: Customization + Hub Publishing
- [ ] Judge prompt presets for GLAM document types
- [ ] Custom prompt and ignore list support
- [ ] `--publish` flag — pushes evaluation to Hub with standardized schema + metadata
- [ ] Define leaderboard dataset schema

### Phase 3: Leaderboard Viewer
- [ ] Viewer on Hub that reads any evaluation dataset matching the schema
- [ ] Approach TBD

### Before Shipping
- [ ] Choose a project name — "ocr-bench" is placeholder. Want something that captures "rankings depend on your documents"

### Phase 4: Blog + Visibility
- [ ] "There Is No Best OCR Model" blog post
- [ ] Cross-link repo ↔ viewer ↔ Hub datasets ↔ blog

### Phase 5: Flagship Run
- [ ] Large-scale run on Britannica, NLS index cards, or BPL catalog

### If project gets traction
- [ ] Consolidate OCR model scripts into this repo + hub-sync to Hub
- [ ] CI/smoke tests
- [ ] Scheduled benchmarking
- [ ] CER/WER metrics alongside VLM judge

## Technical Reference

### ELO
Bradley-Terry, K=32, initial 1500. Position-bias randomization.

### Judge Models
- **Kimi K2.5 (170B, API via Novita)** — best human agreement, default for API
- **Qwen3-VL-30B-A3B (offline vLLM)** — best offline judge
- **7B/8B** — biased toward verbose output, not recommended as primary

### Core Benchmark Models
| Model | Size | Best on |
|-------|------|---------|
| DeepSeek-OCR | 4B | Most consistent across datasets |
| GLM-OCR | 0.9B | Card catalogs |
| LightOnOCR-2 | 1B | Manuscript cards |
| dots.ocr | 1.7B | Historical medical texts |

## Connections

- **uv-scripts/ocr on Hub**: OCR model scripts stay there for now
- **FineCorpus**: OCR quality = training data quality
- **NLS**: Index cards as flagship benchmark dataset
- **Obsidian**: `~/Documents/obsidian/Work/Projects/ocr-benchmark.md`
