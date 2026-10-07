#!/bin/bash
# isa_fn.sh <file.s> <symbol-substring> -- print the body of every kernel whose mangled name contains the substring (label .. .Lfunc_end), minus comments and labels numbering
awk -v pat="$2" '/^[_A-Za-z0-9.]+:/ { name=$0; sub(":.*","",name); if (index(name, pat) && name !~ /^\./) { on=1; print "### " name; next } }
     on && /^\.Lfunc_end/ { on=0 } on { print }' "$1"
