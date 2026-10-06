# rig_copies -- files that live ONLY on the rig, copied here so a lost disk does not lose the service

Copied 2026-10-06 from `~/bench/` on `rome`. **The rig's file stays canonical**; if you change one there, copy it back here
(`scp rome:bench/<file> tools/hot-expert/rig_copies/`) and commit. They are not edited in this repo.

| file | what it is |
|---|---|
| `franken_decode_glm_docker.sh` | launcher of the served GLM-5.3-Flash engine (`franken_dec_glm`) inside the ROCm 7.14 image; `start_franken_glm.sh` runs the engine through it |
| `franken_decode_ds4_docker.sh` | the same for DeepSeek-V4-Flash |
| `franken_decode_docker.sh` | the same for Qwen3.8-Flash-Next |

Also rig-only and NOT copied: `~/bench/franken/glm5/prose8400.txt` (the 8 400-token prose prompt every timing run uses), the model files, the binaries under `~/bench/franken_bin/`.
