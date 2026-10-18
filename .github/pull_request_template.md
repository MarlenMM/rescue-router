## What and why

<!-- What does this change, and why? Link the issue if there is one. -->

## Checks

- [ ] `uv run ruff check src notebook/build_notebook.py release kaggle/phase2 media` and `uv run pytest -q` pass
- [ ] `node --test app/test/*.test.mjs` passes (viewer or router changes)
- [ ] `docs/` rebuilt with `python src/build_viewer.py` (viewer changes)
- [ ] Numbers in the README, model card or data card updated if a result changed
