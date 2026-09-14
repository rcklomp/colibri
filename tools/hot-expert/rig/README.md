# `rig/` — verbatim copies of files that live OUTSIDE the repo on the rome box

These are byte-for-byte copies of scripts the rig runs from `$HOME`, checked in so
they have history, review and blame. **The rig's copy is what actually runs**; this
directory exists so a change to it is visible to the next session.

Keep them byte-identical, so drift is a plain `diff`:

    ssh rome 'cat ~/start_glm53.sh' | diff - tools/hot-expert/rig/start_glm53.sh

`~/bench/restart_gw2.sh` runs exactly that check before it restarts the gateway and
says so when the two differ.

## Why this directory exists

On 2026-09-13 the owner's Open WebUI produced no answer at all. The cause was
`--max-tokens 256` in `~/start_glm53.sh`: `openai_server.py` clamps every request
DOWN to the server cap and Open WebUI sends no `max_tokens` of its own, so 256 was
the ceiling on every reply the box had ever produced, and a tool-calling turn spent
the whole budget opening a `<tool_call>` box it could never close.

It had been there since the oldest surviving copy (2026-09-06 23:05) and **nobody
could say who put it there or why, because the file had no history anywhere.** That
is the gap this directory closes. It is also the only knob in that script that
carried no justifying comment — the comment is now there too.

## Editing rule

Never edit the rig's copy while a copy of it is running: bash reads a script
incrementally and will re-read the changed bytes at its saved offset mid-execution
(hit on 2026-09-13). Stop the gateway first, under the rig lock.

And never put a `#` comment between backslash-continued argument lines: bash joins
them into one logical line, the `#` comments out everything after it, and the
gateway comes up missing arguments while `bash -n` still passes (also hit on
2026-09-13 — the gateway lost `--max-tokens`, `--kv-slots 4` and every
`--allowed-host`). Comments go ABOVE the command.
