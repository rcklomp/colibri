#!/usr/bin/env python3
"""kl_compare.py -- X2: a per-position KL oracle for glm53, beside the
existing greedy-text and last-token-cosine oracles.

Reads two dumps written by `GLM53_LOGIT_DUMP_ALL` (c/glm53.c, main(): a
"GLKD" header -- magic, format version, position count, vocab size, all
uint32 little-endian -- followed by count*vocab float32 logits, row-major by
position) and prints, on one scale:

  - mean KL(ref || cand) over all positions and the max KL + the position
    it occurs at
  - top-1 (argmax) agreement %, with the count and first index of positions
    that disagree
  - the last-position cosine, max-abs and argmax -- the same figure
    `g15_compare.py` / `q4_chain.sh` / CLAUDE.md's oracle already reports,
    so the two scales sit side by side rather than replacing one another

REFUSAL, mirroring `gate_lib.sh`'s `gate_compare` (source of the pattern:
an empty comparison must not print a number that looks like a pass). This
script does not source gate_lib.sh -- that file is bash, this is Python --
but it applies the identical rule and exits 2, printing "REFUSED", when:

  - either file is missing, truncated, or fails the GLKD header check
    (magic / version / a non-positive count or vocab) -- an empty or
    corrupt dump is not a comparison;
  - the two files' position counts differ -- an oracle cannot compare
    position 400 of one run against a run that stopped at 300.

A vocab mismatch is also refused: two dumps from models with different
vocabularies are not the same oracle.

Usage: kl_compare.py <label> <ref.dump> <cand.dump>
"""
import sys, os, struct, math, array

MAGIC = 0x444b4c47  # "GLKD" little-endian, matches c/glm53.c's dump_logits_all


def load(path):
    """Returns (count, vocab, bytes) or None on any refusal condition."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        head = f.read(16)
        if len(head) != 16:
            return None
        magic, version, count, vocab = struct.unpack("<4I", head)
        if magic != MAGIC or version != 1 or count <= 0 or vocab <= 0:
            return None
        data = f.read()
    want = count * vocab * 4
    if len(data) != want:
        return None
    return count, vocab, data


def row(data, vocab, i):
    a = array.array("f")
    a.frombytes(data[i * vocab * 4:(i + 1) * vocab * 4])
    return a


def logsumexp_probs(a):
    mx = max(a)
    ep = array.array("f", (math.exp(x - mx) for x in a))
    z = math.fsum(ep)
    return ep, z, mx


def kl_position(ref_row, cand_row):
    """KL(P_ref || Q_cand) at one position, plus each side's argmax.

    KL(P||Q) = sum_i p_i * (log p_i - log q_i)
             = E_p[ref] - logZ_ref - E_p[cand] + logZ_cand
    avoids ever forming q_i explicitly and reuses the exp() pass that
    logsumexp_probs already did for the softmax normaliser.
    """
    ep_r, z_r, mx_r = logsumexp_probs(ref_row)
    _, z_c, mx_c = logsumexp_probs(cand_row)
    logz_r = math.log(z_r) + mx_r
    logz_c = math.log(z_c) + mx_c
    ep_ref = sum(p * r for p, r in zip(ep_r, ref_row)) / z_r
    ep_cand = sum(p * c for p, c in zip(ep_r, cand_row)) / z_r
    kl = ep_ref - logz_r - ep_cand + logz_c
    ai = ref_row.index(max(ref_row))
    bi = cand_row.index(max(cand_row))
    return kl, ai, bi


def cosine_maxabs(ref_row, cand_row):
    dot = sum(r * c for r, c in zip(ref_row, cand_row))
    sr = sum(r * r for r in ref_row)
    sc = sum(c * c for c in cand_row)
    mx = max(abs(r - c) for r, c in zip(ref_row, cand_row))
    cos = dot / math.sqrt(sr * sc) if sr and sc else float("nan")
    return cos, mx


def main(argv):
    if len(argv) != 4:
        sys.stderr.write("usage: kl_compare.py <label> <ref.dump> <cand.dump>\n")
        return 2
    label, ref_path, cand_path = argv[1], argv[2], argv[3]

    ref = load(ref_path)
    cand = load(cand_path)
    if ref is None or cand is None:
        print("  %-40s REFUSED: empty or corrupt dump (ref=%s cand=%s) -- "
              "an empty comparison is NOT a pass"
              % (label, "ok" if ref else "MISSING/BAD", "ok" if cand else "MISSING/BAD"))
        return 2

    rcount, rvocab, rdata = ref
    ccount, cvocab, cdata = cand
    if rvocab != cvocab:
        print("  %-40s REFUSED: vocab mismatch (ref=%d cand=%d)" % (label, rvocab, cvocab))
        return 2
    if rcount != ccount:
        print("  %-40s REFUSED: position count mismatch (ref=%d cand=%d) -- "
              "position N of one run is not position N of the other"
              % (label, rcount, ccount))
        return 2

    n = rcount
    kls = []
    agree = 0
    first_disagree = None
    for i in range(n):
        rr = row(rdata, rvocab, i)
        cr = row(cdata, cvocab, i)
        kl, ai, bi = kl_position(rr, cr)
        kls.append(kl)
        if ai == bi:
            agree += 1
        elif first_disagree is None:
            first_disagree = i

    mean_kl = sum(kls) / n
    max_kl = max(kls)
    max_pos = kls.index(max_kl)
    top1_pct = 100.0 * agree / n

    last_r = row(rdata, rvocab, n - 1)
    last_c = row(cdata, cvocab, n - 1)
    cos, maxabs = cosine_maxabs(last_r, last_c)

    print("  %s  (%d positions, vocab %d)" % (label, n, rvocab))
    print("     mean KL(ref||cand)  : %.6g" % mean_kl)
    print("     max  KL(ref||cand)  : %.6g  at position %d" % (max_kl, max_pos))
    print("     top-1 agreement     : %.2f%%  (%d/%d)%s"
          % (top1_pct, agree, n,
             "" if agree == n else "  first disagreement at position %d" % first_disagree))
    print("     last-position cosine: %.9f  maxabs %.6g" % (cos, maxabs))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
