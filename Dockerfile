FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src/ ./src/

RUN uv pip install --system --no-cache ".[viewer]"

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

ENV REPOS="davanstrien/bpl-ocr-bench-results"

EXPOSE 7860
CMD ["python", "-m", "ocr_bench.space"]
