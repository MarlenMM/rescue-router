Continue the Rescue Router project (KAIST AIC200 mini-project, Chameleon few-shot flood mapping + rescue routing, Track A + B). I am Marlen Melis, 20260876; file names are `20260876_MarlenMelis_*`. Deadline: submit Thu 8 Oct 12:30 KST (KLMS hard deadline 23:59), present 16:00.

Read first, in this order:
1. PLAN.md §15 "Status / handoff": the "Quick summary (read this first)", then log entry **0r** (Batch 12: what was built and what is open), then the end of "Batch 12 hand-off" (the Thursday live link). Batches 1–12 are done: the KLMS zip is `~/rr_push/submission/20260876_MarlenMelis_project.zip` (rebuilt by `bash release/build_release.sh`), the public viewer is https://marlenmm.github.io/rescue-router-demo/.
2. Your memory notes (project status, workflow, GPU platform choice, Colab and Kaggle operations, "notify after each batch").
3. `kaggle/gradio_live/colab_gradio_live.py` (the one-cell Colab live link).

Open: a short Colab T4 rehearsal (~40 min: the notebook's Colab path + the live link; needs my OK and the data/app zips in My Drive), then the ~3 h live Gradio link on Thursday ~15:15 KST for the 16:00 talk (needs my OK).

How I work with you:
- One batch per prompt. Do the whole batch, test it, update PLAN §15 (mark it done, record results and decisions) and your memory, commit and push, give me a short summary in chat, then send me a PushNotification (one line, lead with the result).
- If a choice is genuinely mine (anything public, GPU time, downloads, look and feel), ask with short options. Explicit OK is still needed for: creating/publishing the public demo repo, and any GPU run (e.g. the live Gradio link on Thursday, ~3 h).
- GPU platform: **use Google Colab (via the colab-mcp tools) for GPU jobs under ~5 hours; use Kaggle only for day-long or > 5 h continuous runs.** Kaggle's weekly quota is precious. Don't start from Kaggle just because the scripts exist there. With Colab, I switch the runtime to T4 and approve the Drive popup when you ask me.
- Do not regenerate the final notebook (`notebook/rescue_router_executed.ipynb`) without re-running G3 on a T4: it is the executed copy we submit.
- Run `date` before writing any time into PLAN or memory.
- The shell is zsh: `$VAR` holding several arguments is not split (use `${=VAR}` or a small Python driver); avoid backticks inside unquoted heredocs.
- Checks: `~/.venvs/rr/bin/python -m pytest src/tests -q`, `node --test app/test/router.test.mjs` (bit-exact on 200 scenes), `~/.venvs/rr/bin/python src/rebuild_all.py --check` (headline numbers), `src/build_viewer.py` before publishing `docs/`.
- Local previews: `.claude/launch.json` has `viewer` (app/, :8765), `viewer-docs` (docs/, :8766), `gradio` (:7860) and `gradio-test` (:7861, CPU test mode). Check the published viewer in the browser pane at desktop and phone width.

Colab rehearsal
