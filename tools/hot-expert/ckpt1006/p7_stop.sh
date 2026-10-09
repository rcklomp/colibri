#!/bin/bash
# Stop the gateway and WAIT for its engine to actually die. A plain
# "pkill -9 -x glm53; sleep 3" is not enough: unmapping ~180 GiB takes longer
# than that, and the next harness refuses on the corpse (p7 smoke, 2026-09-07).
pkill -f "openai_[s]erver.py"
for i in $(seq 1 30); do pgrep -f "openai_[s]erver.py" >/dev/null || break; sleep 1; done
pkill -9 -x glm53 2>/dev/null
for i in $(seq 1 120); do pgrep -x glm53 >/dev/null || break; sleep 1; done
echo "stopped: gateway=$(pgrep -f "openai_[s]erver.py" | wc -l) engine=$(pgrep -x glm53 | wc -l)"
pgrep -x glm53 >/dev/null && exit 2
exit 0
