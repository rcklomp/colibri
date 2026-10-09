#!/bin/bash
# PF0 depth curve: token IDs of the first 420 KB of the measurement record (the needle test's own corpus), [gMASK]<sop> first. No GPU device is given to the container.
set -u
B=$HOME/bench/franken/glm5; O=$B/rec_depth; mkdir -p $O
M=$HOME/models/GLM-5.3-Flash/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
head -c 420000 $HOME/src/colibri/tools/hot-expert/ROME-3x7900XTX-2026-09-04.md > $O/rec420k.txt
sha256sum $O/rec420k.txt | cut -c1-16
docker run --rm -e LD_LIBRARY_PATH=/opt/rocm/lib:/home/ronald/src/llama-glm53/build-hip/bin -e HOME=/home/ronald -v /home/ronald:/home/ronald -w $O \
  rocm/dev-ubuntu-24.04:7.14.0-full /home/ronald/src/llama-glm53/build-hip/bin/llama-tokenize -m $M --ids --log-disable -f $O/rec420k.txt > $O/rec420k.ids.raw 2> $O/tok.err
rc=$?; echo "tokenize rc=$rc"; [ $rc -eq 0 ] || { echo FAILED; tail -3 $O/tok.err; exit 1; }; head -c 200 $O/rec420k.ids.raw; echo; tail -3 $O/tok.err | cut -c1-200
python3 -I - <<'PY'
import re, os
O = os.path.expanduser('~/bench/franken/glm5/rec_depth')
s = open(O + '/rec420k.ids.raw').read()
ids = [int(x) for x in re.findall(r'\d+', s)]
print('ids', len(ids), 'first', ids[:4])
assert len(ids) > 70000, 'too few ids'
if ids[:2] != [154822, 154824]:
    ids = [154822, 154824] + ids
    print('prefixed gMASK/sop; first', ids[:4])
open(O + '/rec_ids.txt', 'w').write(' '.join(map(str, ids)) + '\n')
print('wrote', len(ids))
PY
echo "tok_rec done $(date -Is)"
