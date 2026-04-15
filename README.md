# humatheque-vlm-bench

`humatheque-vlm-bench` is a Hugging Face Hub-native benchmark toolkit for **vision-language metadata extraction** and **pairwise LLM-as-a-judge ranking**.

This project is a Humathèque-focused fork of the original `ocr-bench` by Daniel van Strien.

This repository is currently configured for the Humathèque thesis-cover task:

- **Dataset**: `Geraldine/humatheque-vlm-sudoc-grounded`
- **Input image column**: `image_uri`
- **Ground truth JSON column**: `sudoc_record_templated`
- **Default extraction models**:
  - `Qwen/Qwen3-VL-4B-Instruct`
  - `nanonets/Nanonets-OCR2-3B`
  - `google/gemma-4-E4B-it`

---

## What’s new in this setup

Compared to the original OCR-centric flow, this configuration adds:

1. **Task-configured metadata extraction prompt**
   - Full JSON schema-like instruction with controlled vocabularies for `degree_type` and `discipline`.
2. **Standard reference-based evaluation**
   - Field-level scoring against `sudoc_record_templated` with:
     - exact match for selected fields (`defense_year`, `degree_type`, `language`)
     - fuzzy matching for text fields
     - list matching for multi-name fields
     - `jury_global` consolidation metric
     - per-field precision / recall / F1 and global mean F1
3. **Pairwise LLM-as-a-judge ranking**
   - Existing pairwise comparison + ELO ranking pipeline remains available.

---

## Install

```bash
git clone https://github.com/<your-org>/humatheque-vlm-bench.git
cd humatheque-vlm-bench
uv pip install -e .[viewer]
```

If `humatheque-vlm-bench` is still "command not found", refresh the editable install:

```bash
uv pip install -e .
```

Requires:

- Python >= 3.11
- Hugging Face token (`HF_TOKEN`)

---

## End-to-end workflow

### 1) Run model inference jobs on HF Jobs

```bash
humatheque-vlm-bench run Geraldine/humatheque-vlm-sudoc-grounded <your-output-dataset> --max-samples 50
```

By default, this launches the three configured VLMs.  
You can list or override models:

```bash
humatheque-vlm-bench run in out --list-models
humatheque-vlm-bench run in out --models qwen3-vl-4b-instruct gemma-4-e4b-it
humatheque-vlm-bench run in out --prompt "Extract thesis metadata as JSON"
```
By default, scripts run with their own prompt defaults. Use `--prompt` to force a custom prompt.

### 2) Run evaluation and ranking

```bash
humatheque-vlm-bench judge <your-output-dataset> --ground-truth-column sudoc_record_templated
```

`judge` now does **both**:

- **Standard evaluation table** (global F1 + jury global F1 per model)
- **Pairwise LLM-as-a-judge** comparisons + ELO leaderboard

You can choose a different judge model or jury:

```bash
humatheque-vlm-bench judge <your-output-dataset> \
  --model novita:Qwen/Qwen3.5-35B-A3B \
  --model together:meta-llama/Llama-3.3-70B-Instruct-Turbo
```

### 3) Browse results locally

```bash
humatheque-vlm-bench view <your-output-dataset>-results
```

### 4) (Optional) Publish viewer as a Space

```bash
humatheque-vlm-bench publish <your-output-dataset>-results
```

---

## Standard evaluation details

The standard benchmark compares each model output JSON to `sudoc_record_templated`:

- Parses prediction + GT JSON records
- Computes TP / FP / FN per field
- Aggregates to precision / recall / F1
- Reports:
  - per-field metrics
  - `jury_global` metrics (combined committee names)
  - model-level `global_f1` (mean F1 across fields)

This gives a reproducible, reference-based signal that complements LLM-as-a-judge ranking.

---

## Useful commands

```bash
# Dry run jobs
humatheque-vlm-bench run Geraldine/humatheque-vlm-sudoc-grounded out --dry-run

# Evaluate only a subset of rows
humatheque-vlm-bench judge out --max-samples 25

# Disable adaptive pairwise stopping
humatheque-vlm-bench judge out --no-adaptive

# Re-judge from scratch
humatheque-vlm-bench judge out --full-rejudge
```

---

## Notes

- The standard evaluation implementation is intended for this metadata extraction benchmark and can be extended for custom weighting or additional schema rules.
- For best reproducibility, keep model outputs as valid JSON objects aligned with the configured schema fields.
