# STAT 563 HW2

Python-based workflows (managed with `uv`) for STAT 563 Project #2. It mirrors the
original MATLAB modules and generates all figures/data needed for the final LaTeX
write-up.

## Quickstart

```bash
uv sync                          # install deps
uv run python scripts/run_logistic_pipeline.py
uv run python scripts/run_correlation_pipeline.py
```

Outputs land in `outputs/data/` and `outputs/figures/`. The original assignment
materials remain under `reference_materials/`.
