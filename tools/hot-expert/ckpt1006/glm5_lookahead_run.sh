#!/bin/bash
# glm5_lookahead_run.sh -- wrapper so run_chain.sh (rig lock) can launch the lookahead probe chain from its worktree.
mkdir -p ~/bench/franken/glm5/lookahead
cd ~/src/franken-engine-lookahead/franken/decode || exit 2
GLM5_GPU_OK=1 ./glm5_lookahead_chain.sh run 2>&1 | tee ~/bench/franken/glm5/lookahead/chain.txt
exit ${PIPESTATUS[0]}
