Continue the Rescue Router project (KAIST AIC200 mini-project, Chameleon few-shot flood mapping + rescue routing, Track A + B). I am Marlen Melis, 20260876; file names are `20260876_MarlenMelis_*`. Deadline: submit Thu 8 Oct 12:30 KST (KLMS hard deadline 23:59), present 16:00.

Read first, in this order:
1. PLAN.md §15 "Status / handoff": the "Quick summary (read this first)", then the **"Batch 13: model improvements before the deadline"** block (the 36 ideas I1–I37, the adoption rule, the GPU budget and batches 13.1–13.6), then log entries 0r–0t (Batch 12, Colab check, wording fixes).
2. Your memory notes (project status, workflow, GPU platform choice, Colab and Kaggle operations, "notify after each batch").
3. The course's rules: `README_starter.md` ("IMPORTANT NOTES", "What to Submit").

State: Batches 1–12 are done. The submission is built and verified: `~/rr_push/submission/20260876_MarlenMelis_project.zip` (built 7 Oct 16:01; it is the fallback and must stay untouched until a better one passes). Public viewer https://marlenmm.github.io/rescue-router-demo/. The notebook passed on Colab (30 min, 0 errors). Slide 3 changed in the deck; the PPTX in the zip is still the old one until I re-export it.

Batch 13 = improve the FloodNet 50-shot model before the deadline: try every idea I1–I37 except I26, adopt whatever improves or does not hurt (rule in PLAN), always apply the efficiency items I33–I37, and switch the whole submission to the new model by ~11:00 Thu only if it passes the full-test check. Work batch by batch (13.1 → 13.6), running independent lanes in parallel.

GPU: Kaggle ~12–13 h left this week (check the quota first), Colab ~1 h today. You have my OK to use up to that budget for Batch 13 (Kaggle and Colab) without asking again for each run; still ask before publishing anything public (Pages republish) and before replacing the submission zip. For Colab I switch the runtime to T4 and approve the Drive popup when you ask. Colab's MCP tools drop their connection and time out on cells over ~30 min: run long work as background processes in the kernel and poll with short cells.

How I work with you:
- Do the whole batch, test it, update PLAN §15 (mark it done, record results and decisions, log every variant with its val30 numbers) and your memory, commit and push, give me a short summary in chat, then send me a PushNotification (one line, lead with the result).
- If a choice is genuinely mine (anything public, look and feel, replacing the zip), ask with short options.
- Do not regenerate the final notebook (`notebook/rescue_router_executed.ipynb`) without re-running it on a T4.
- Run `date` before writing any time into PLAN or memory.
- The shell is zsh: `$VAR` holding several arguments is not split (use `${=VAR}` or a small Python driver); avoid backticks inside unquoted heredocs.
- Checks: `~/.venvs/rr/bin/python -m pytest src/tests -q`, `node --test app/test/router.test.mjs` (bit-exact on 200 scenes), `~/.venvs/rr/bin/python src/rebuild_all.py --check` (headline numbers), `src/build_viewer.py` before publishing `docs/`. New options must keep today's defaults bit-exact.
- Local previews: `.claude/launch.json` has `viewer` (app/, :8765), `viewer-docs` (docs/, :8766), `gradio` (:7860) and `gradio-test` (:7861, CPU test mode).

Batch 13.1
