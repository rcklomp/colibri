# archive/

Things that were cleaned off the rig but are kept in git so no recipe is lost.

- `rig-bench-scripts-2026-10-09.tgz` -- the 144 one-off chain / queue / smoke / report scripts that sat loose in `~/bench` on the rig (September to early October 2026: the P*, Q*, M*, F11, glm5* chains and similar, 680 KB unpacked) and existed in no repository. Secret-scanned before archiving (only `127.0.0.1` appears). They were deleted from the rig on 2026-10-09 (the clean-up that made `closeout_check.sh --full` FAIL on stray scripts). To read one: `tar xzOf archive/rig-bench-scripts-2026-10-09.tgz <name>`; to restore: `tar xzf archive/rig-bench-scripts-2026-10-09.tgz -C ~/bench <name>`. Anything worth reusing belongs in `ckpt1006/` as a tested recipe, not back in `~/bench`.
