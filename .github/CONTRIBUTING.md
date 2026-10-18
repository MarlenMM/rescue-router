# Contributing

Thanks for taking the time. Bug reports, questions about the method and small fixes are all welcome; for anything
larger, please open an issue first so we can agree on the approach.

## Set-up

```bash
uv sync --locked                 # Python 3.12, CPU environment from uv.lock
uv run pre-commit install        # ruff + the fast tests on every commit, as in CI
```

Node.js ≥ 20 runs the viewer's tests. Data and prediction caches are not in the repository; tests that need them skip
themselves (see the README's *Getting started*).

## Before you open a pull request

```bash
uv run ruff check src notebook/build_notebook.py release kaggle/phase2 media
uv run pytest -q
node --test app/test/*.test.mjs
```

A few rules keep the published numbers trustworthy:

- **The router has two implementations that must agree bit for bit**: `src/router.py` (the reference) and
  `app/js/router.js` (the viewer). A change to one needs the same change to the other; `app/test/router.test.mjs`
  checks them against each other on all 400 scene × model pairs.
- **`docs/` is generated.** Edit `app/`, then run `python src/build_viewer.py`; CI checks that `docs/` is the build of
  `app/`.
- **Every interface string needs its Korean translation** in `app/js/i18n.js` (`app/test/i18n.test.mjs` checks this).
- **The study protocol is fixed.** `experiments/phase2/PROTOCOL.md` was committed before its runs and is not edited
  afterwards; a new experiment gets its own written plan before it runs.
- **Choices are made on the tuning photos, never on the test set**, and a result is reported whether or not it helps.

## Style

Python follows the existing code: ruff with the rules in `pyproject.toml`, lines up to 120 characters, short
docstrings that say what a function returns and why. Commit messages say what changed and why, in one line where
possible.
