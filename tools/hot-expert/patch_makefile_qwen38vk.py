p = "/home/ronald/src/colibri/c/Makefile"
s = open(p).read()
anchor = ("qwen38$(EXE): qwen38.c cli_args.h qwen38_core.h qwen38_nfc.h qwen38_nfc_tables.h "
          "st.h json.h compat.h quant.h route_trace.h tok.h tok_unicode.h tok_unicode_o200k.h serve_codec.h\n"
          "\t$(CC) $(NOCUDA_CFLAGS) qwen38.c -o qwen38$(EXE) $(NOCUDA_LDFLAGS)\n")
assert s.count(anchor) == 1, "anchor miss/not-unique: %d" % s.count(anchor)
addition = anchor + """
# Opt-in GPU-accelerated variant (VK=1 required): adds the persistent fmt=8
# expert cache from tools/hot-expert/patch_qwen38_gputier.py. The default
# qwen38 target above stays exactly as documented -- intentionally CPU-only.
qwen38-vk$(EXE): qwen38.c cli_args.h qwen38_core.h qwen38_nfc.h qwen38_nfc_tables.h st.h json.h compat.h quant.h route_trace.h tok.h tok_unicode.h tok_unicode_o200k.h serve_codec.h backend_vulkan.h $(VK_OBJ) $(VK_SPV)
\t$(CC) $(NOCUDA_CFLAGS) -DQ38_VK_TIER qwen38.c $(VK_OBJ) -o qwen38-vk$(EXE) $(NOCUDA_LDFLAGS)
"""
s = s.replace(anchor, addition)
open(p, "w").write(s)
print("MAKEFILE TARGET ADDED")
