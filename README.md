# ocr-bench

**There is no single best OCR model.** Rankings change by document type — manuscript cards, printed books, and historical texts all produce different winners. ocr-bench creates per-collection leaderboards using a VLM-as-judge approach so you can find what works best for *your* documents.

## Quickstart

```bash
pip install ocr-bench[viewer]

# 1. Run OCR models on your dataset
ocr-bench run <input-dataset> <output-repo> --max-samples 50

# 2. Judge outputs pairwise with a VLM
ocr-bench judge <output-repo>

# 3. Browse results + validate
ocr-bench view <output-repo>-results
```

## Install

```bash
pip install ocr-bench            # Core (run + judge)
pip install ocr-bench[viewer]    # With web UI (for ocr-bench view)
```

Requires Python >= 3.11.

## How it works

**`ocr-bench run`** launches OCR models on your dataset via [HF Jobs](https://huggingface.co/docs/hub/jobs). Each model writes its output as a config or PR branch on the same Hub dataset, keeping everything together.

**`ocr-bench judge`** runs pairwise comparisons using a VLM judge (default: [Kimi K2.5](https://huggingface.co/moonshotai/Kimi-K2.5)). For each document, the judge sees the original image and two OCR outputs (anonymized as A/B) and picks the better one. Results are fit to a [Bradley-Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) to produce ELO ratings with bootstrap 95% confidence intervals.

**`ocr-bench view`** serves a local web UI (FastAPI + HTMX) with a leaderboard, comparison browser, and human validation. Keyboard-first: `a`/`b`/`t` to vote, `r` to reveal the judge verdict, arrow keys to navigate.

## Example results

Rankings flip depending on the collection:

| Model | BPL card catalog | Britannica 1771 |
|-------|:---:|:---:|
| LightOnOCR-2 (1B) | **#1** (1559) | #2 (1703) |
| GLM-OCR (0.9B) | #2 (1535) | **#1** (1779) |
| DeepSeek-OCR (4B) | #4 (1452) | #3 (1362) |
| dots.ocr (1.7B) | #3 (1453) | #4 (1156) |

ELO scores in parentheses. 95% CIs available in the full results.

Browse these on the Hub:
- [davanstrien/bpl-ocr-bench-results](https://huggingface.co/datasets/davanstrien/bpl-ocr-bench-results) — Boston Public Library card catalog, 4 models, 150 samples
- [davanstrien/ocr-bench-britannica-results](https://huggingface.co/datasets/davanstrien/ocr-bench-britannica-results) — Encyclopaedia Britannica 1771, 4 models, 50 samples

## Live demo

Try the viewer without installing anything:

[**ocr-bench-viewer** on Hugging Face Spaces](https://huggingface.co/spaces/davanstrien/ocr-bench-viewer)

## Key features

- **Smart defaults** — `ocr-bench judge <dataset>` needs zero flags; auto-detects model configs, derives results repo name
- **Incremental judging** — loads existing comparisons, skips judged pairs, judges only new ones
- **Adaptive stopping** — stops early when rankings are statistically resolved (on by default)
- **Concurrent judge calls** — `--concurrency N` for faster evaluation
- **Multi-judge jury mode** — pass multiple `--model` flags to average across judges
- **Keyboard-first viewer** — browse comparisons, vote, reveal verdicts, all without touching the mouse

## Status

This is a working proof of concept. The core pipeline (run, judge, view) is functional and tested (220+ tests). It's not polished production software — expect rough edges.
