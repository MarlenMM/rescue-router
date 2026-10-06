Continue the Rescue Router project (KAIST AIC200 mini-project, Chameleon few-shot flood mapping + rescue routing, Track A + B). I am Marlen Melis, 20260876; file names are `20260876_MarlenMelis_*`. Deadline: submit Thu 8 Oct 12:30 KST (KLMS hard deadline 23:59), present 16:00.

Read first, in this order:
1. PLAN.md §15 "Status / handoff": the "Quick summary (read this first)", then **"Batch 12 hand-off"** (what to ask me, what to build, how to check it), then the "Improvement batches" list. Batches 1–11 are done: product, evaluation, final notebook (G3 passed on a fresh T4), write-up PDF, slides (Claude Slides artifact https://claude.ai/artifact/YYgLzsuLXTeRiQmWUtpyJw).
2. Your memory notes (project status, workflow, Kaggle operations, "notify after each batch").
3. The course's rules for the submission: `README_starter.md` ("IMPORTANT NOTES", "What to Submit").

Batch 12 = packaging + release: the data zip, the app zip, the project zip with exactly one .ipynb, the public demo on GitHub Pages, and the checklist in PLAN §12. Start by asking me (short options, one question set): the data-zip size choice (363 MB vs our 300 MB target), the OK to create and publish `MarlenMM/rescue-router-demo`, and whether the slides PDF and the recording are ready.

How I work with you:
- One batch per prompt. Do the whole batch, test it, update PLAN §15 (mark it done, record results and decisions) and your memory, commit and push, give me a short summary in chat, then send me a PushNotification (one line, lead with the result).
- If a choice is genuinely mine (anything public, GPU time, downloads, look and feel), ask with short options. Explicit OK is still needed for: creating/publishing the public demo repo, and any GPU run (e.g. the live Gradio link on Thursday, ~3 h).
- GPU platform: **use Google Colab (via the colab-mcp tools) for GPU jobs under ~5 hours; use Kaggle only for day-long or > 5 h continuous runs.** Kaggle's weekly quota is precious. Don't start from Kaggle just because the scripts exist there. With Colab, I switch the runtime to T4 and approve the Drive popup when you ask me.
- Do not regenerate the final notebook (`notebook/rescue_router_executed.ipynb`) without re-running G3 on a T4: it is the executed copy we submit.
- Run `date` before writing any time into PLAN or memory.
- The shell is zsh: `$VAR` holding several arguments is not split (use `${=VAR}` or a small Python driver); avoid backticks inside unquoted heredocs.
- Checks: `~/.venvs/rr/bin/python -m pytest src/tests -q`, `node --test app/test/router.test.mjs` (bit-exact on 200 scenes), `~/.venvs/rr/bin/python src/rebuild_all.py --check` (headline numbers), `src/build_viewer.py` before publishing `docs/`.
- Local previews: `.claude/launch.json` has `viewer` (app/, :8765), `viewer-docs` (docs/, :8766), `gradio` (:7860) and `gradio-test` (:7861, CPU test mode). Check the published viewer in the browser pane at desktop and phone width.

Batch 12
