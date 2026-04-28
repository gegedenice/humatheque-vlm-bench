"""Tests for task-config prompt builders."""

from ocr_bench.task_config import (
    DISSERTATION_DEGREE_TYPE_VALUES,
    THESIS_DEGREE_TYPE_VALUES,
    build_eval_prompt,
)


def test_build_eval_prompt_thesis_values():
    prompt = build_eval_prompt("thesis")
    assert "graduate thesis" in prompt
    for value in THESIS_DEGREE_TYPE_VALUES:
        assert value in prompt


def test_build_eval_prompt_memoire_values():
    prompt = build_eval_prompt("memoire")
    assert "graduate dissertation" in prompt
    for value in DISSERTATION_DEGREE_TYPE_VALUES:
        assert value in prompt
