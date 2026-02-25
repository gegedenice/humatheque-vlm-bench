"""HF Space entry point for ocr-bench viewer."""

import os

import uvicorn

from ocr_bench.web import create_app


def main():
    repos = os.environ.get("REPOS", "davanstrien/bpl-ocr-bench-results")
    repo_id = repos.split(",")[0].strip()
    app = create_app(repo_id, output_path="/tmp/annotations.json")
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
