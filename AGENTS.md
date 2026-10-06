# For any coding agent working in this repository

This is the fork `rcklomp/colibri`, worked on a rig called "rome". Before
forming any plan, read, in this order:

0. Run `tools/hot-expert/doc_currency.sh` (seconds, no rig) -- it fails when the docs are behind the tree.
1. `CLAUDE.md` — the rules and traps of this machine. They apply to every
   agent, not only Claude. Its "Read first" section lists the rest.
2. `tools/hot-expert/HANDOFF-2026-10-06b.md` — the single current entry point (state, repos and accounts, the service and its traps, open tasks, recipes). The older
   `PROJECT-HANDOFF-2026-09-20.md` (the whole project in one file, as of 2026-09-20) is history, but **its section 0 lists what the previous session got wrong and what it
   cost the owner (an hour of his daily service down, a wasted item, his plan's
   usage limit). Do not repeat those mistakes.**

The three that matter most:

- The gateway on port 8081 is the owner's daily service (currently OFF behind `~/bench/.dev_reserved`; see the 06b handoff §3 before touching that flag). Stop it only inside a
  chain under the rig lock, and **after any chain has fully exited, send one
  real chat request** — `/v1/models` = 200 and `pgrep -x glm53` both report a
  dead engine as alive.
- **Step 0 of any performance item is a measurement, never arithmetic.**
  Measure the bucket, re-derive the gate, then build.
- The owner is not a developer and is on a fixed plan. One decision with its
  reason, then the work, then the number. Use the cheapest model that can do
  the job.
