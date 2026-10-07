#!/bin/bash
# isa_mix.sh <fn.s> -- per kernel (### lines from isa_fn.sh): opcode histogram (first token of each instruction line)
awk '/^### / { k=$2; next } /^\t[a-z_0-9]+/ { op=$1; c[k" "op]++ } END { for (x in c) print x, c[x] }' "$1" | sort
