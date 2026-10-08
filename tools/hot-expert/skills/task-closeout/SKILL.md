---
name: task-closeout
description: Run at the END of every finished task or work stretch in the colibri / franken-engine / rome-rig project, BEFORE the final report, without being asked -- and whenever the user asks "did you do your homework", "are the repos synced", "check the governance docs". Verifies repos synced (Mac, Gitea, rig), trees clean, governance docs current (plan rev, record, handoff, CLAUDE.md), rig state, installed binary named in the docs, memory updated; fixes what it finds; reports plainly what is verified and what is not. Use after any task that changed code, docs, scripts, the rig or the installed binary.
---

# Task closeout

The owner had to ask for this after every completed task (2026-10-07, angrily). It is part of the task: do it unprompted, before the final message, never ask permission for it.

## Steps

1. **Run the check:** `bash ~/Projects/colibri/tools/hot-expert/closeout_check.sh --full` (read-only except a `git fetch`; ~10 s). It checks repo sync (Mac / Gitea / rig, both repos), clean trees, `doc_currency.sh`, that the handoff names the installed serving binary, rig idle/busy, drifted `~/bench` script copies, the installed skill and hook, the memory index.
2. **Fix every FAIL.** Recipes: handoff 06b section 2 (land and relay). colibri: commit on `hot-expert-tier`, `git push origin hot-expert-tier`, `git push rome hot-expert-tier:refs/heads/p0-sync`, on the rig `git -C ~/src/colibri merge --ff-only p0-sync && git -C ~/src/colibri branch -d p0-sync`. franken-engine: land from the Mac clone (`git merge --ff-only <branch>`, push as `claude-bot` with the credential-store helper, never print the credentials), then `merge --ff-only` on the rig. Commits end with the attribution lines the session's system reminder gives. **The public GitHub fork is pushed only with the owner's explicit go-ahead after a secret scan: report its lag, never push it.** Decide each WARN: fix it or name it in the report.
3. **Governance read-through** (what the script cannot judge):
   - the plan (`tools/hot-expert/FRANKEN-ENGINE-PLAN-2026-09-15.md`) has a Rev entry for what changed (look up the top `Rev N` first; CLAUDE.md and the handoff must name it: `doc_currency.sh` checks that);
   - the record (`ROME-3x7900XTX-2026-09-04.md`) has a section at its END with the numbers and which oracle/gate passed; a claim that was later found wrong is corrected in an erratum, not silently edited;
   - the handoff `HANDOFF-2026-10-06b.md`: section 0 / 1 (state, installed sha, rollbacks), 4 (open tasks), 5 (recipes), 6 (traps) are current; a cold reader must not be misled;
   - `CLAUDE.md` pointers (rev, recipes) are current;
   - **the action list** (`tools/hot-expert/ACTION-LIST.md`): the rows this task touched have the right status (`doing` while a rig job runs, `done` / `dropped` with a Result pointer and a line in the Done log when the result is in the record), a gate that was decided has released or dropped its `gated` rows, a newly found item has a plan row first and then a list row, the 'Last updated' line is today's; `doc_currency.sh` checks the mechanics, you check that the order is still the right one;
   - memory (`~/.claude/projects/-Users-ronald-Projects/memory/`): the state note says what a future session needs; a new owner correction or confirmed approach is a feedback memory; nothing the repo already records, no passwords or secrets, no volatile detail beyond pointers.
4. **Clean up what you created:** rig worktrees, one-off sequencers, temp scripts, background watchers that are no longer needed (a rig job still running needs a watcher armed and must be reported).
5. **Re-run the check until 0 FAIL.**
6. **Final report:** short, plain English, few numbers. Say what was verified and what was NOT (for example "not run through the serve path"), what needs the owner's decision (GitHub pushes, hardware), and the one thing to do next.

## Automation

A Stop hook (`closeout_stop_hook.sh`, registered in `~/.claude/settings.json`) runs the quick check when a turn ends and blocks the stop when a repo is unsynced or dirty and the rig is idle. It never loops, never blocks while a rig job runs, and ignores other projects. If it blocks, this skill is the way out: fix, then finish.

Canonical copy of this file: `~/Projects/colibri/tools/hot-expert/skills/task-closeout/SKILL.md`; the installed copy is `~/.claude/skills/task-closeout/SKILL.md` (the check warns when they differ).
