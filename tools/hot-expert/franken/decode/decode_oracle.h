// tools/hot-expert/franken/decode/decode_oracle.h
//
// Reads the dump `franken_oracle_dump` writes (L0-STEP2-BRIEF-2026-09-22.md,
// deliverable A) and compares this engine's taps against it.
//
// Format, as deliverable A actually writes it (read off
// ~/bench/franken/oracle/index.txt, 2026-09-22): each line is
// `<key> <il> <ne0> <ne1> <ne2> <type>` and the file is `<dir>/<key>.f32`.
// The KEY already carries the layer ("hc_norm-3") and, where the reference
// cb's a name twice in one layer, a disambiguating ".2" ("hc_norm-3.2"); a
// graph-level tap has no suffix at all ("hc_init", "ple_embd") and its il is
// -1. The payload is the LAST slice along the tensor's slowest NON-UNIT axis
// (verified against the files, not assumed):
//
//   ne2 > 1   ->  ne0*ne1 floats, the last ne2 slice.  hc_combine-0
//                 [2560,4,6] is 10240 floats = the last token's four hc
//                 streams; state_predelta-0 [128,128,48] is 16384 floats,
//                 and its ne2 is the HEAD axis, so that file is head 47's
//                 128x128 state, not a token.
//   ne2 == 1  ->  ne0 floats, the last ne1 column.  indexer_score-3
//                 [64,6,1] is 64 floats = all 64 block scores for the last
//                 token, but indexer_k_pooled-3 [128,64,1] is 128 floats =
//                 the last BLOCK's pooled key, and which block that is
//                 depends on llama.cpp's PADDED n_kv (256 cells for a
//                 6-token prompt). See the incomparable list in the .cpp.
//
// decode_graph.cpp builds the same keys and taps the same slices.
//
// The comparison is deliberately not "read the numbers yourself": it follows
// gate_lib.sh's rule (CLAUDE.md, "How a change is measured") that a
// comparison whose two sides are not both present is a REFUSAL, not a pass.
// Two absent arrays diffed against each other print IDENTICAL, and that is
// how a gate once reported a passing oracle from two runs that had produced
// no output at all. Here: a tap with no oracle file is MISSING (reported,
// never counted as a pass); an oracle file with no tap is UNCHECKED; a
// zero-norm pair is a FAIL.

#pragma once

#include <map>
#include <string>
#include <vector>

#include "decode_graph.h"

namespace fk {

struct OracleEntry {
    std::string key;      // the index's first field, which is also the file name
    int         il = -1;
    int64_t     ne0 = 0, ne1 = 0, ne2 = 0;
    std::string type;
    std::string path;
};

class Oracle {
public:
    // Returns false (and says why) if the directory or its index is absent --
    // the brief allows the comparison to be reported as PENDING in that case.
    bool load(const std::string & dir, std::string & why);

    bool loaded() const { return loaded_; }
    const std::string & dir() const { return dir_; }
    // Whether the dump carries `key` (a readable file). The DeepSeek-V4 CLI
    // uses it for its own required-point check (ds4_graph.cpp).
    bool has(const std::string & key) const { return entries_.count(key) != 0; }
    size_t size() const { return entries_.size(); }

    // Compares every tap the recorder holds against the entry of the same
    // (name, il). Prints one `oracle <name>-<il> ...` line per compared point.
    // `pass_layers` are the layers whose l_last / hc_combine must also pass
    // (the brief's bar: it is the layer OUTPUT that has to match, not just
    // the pointwise taps on the way there).
    // Returns true when every compared point met `min_cos`.
    bool compare(const Recorder & rec, const std::vector<int> & pass_layers,
                 double min_cos, FILE * out);

private:
    bool read_file(const OracleEntry & e, std::vector<float> & out) const;

    bool        loaded_ = false;
    std::string dir_;
    std::map<std::string, OracleEntry> entries_;
};

// ------------------------------------------------- the step-3 oracles ----
//
// Taps prove the arithmetic; these two prove the MODEL. Step 2 established
// that the per-point cosines stop being a test once the MoE router's
// top-10-of-512 selection diverges, so step 3 adds the two checks that
// survive that: what the model actually SAYS, and which experts it picked.

// (b) greedy ids, one per line, against llama.cpp at temperature 0. `#` and
// blank lines are ignored so the dump tool may annotate the file.
bool load_expected_ids(const std::string & path, std::vector<int> & out, std::string & why);
// -1 when the common prefix agrees; otherwise the first differing position.
int  first_mismatch(const std::vector<int> & mine, const std::vector<int> & ref);

// (c) the routing oracle the record now names. `<dir>/moe_ids.txt` holds
// lines `il pos: id id ...`; the score is the mean over positions of
// |A n B| / n_expert_used, per layer. It is a SET overlap, not an ordered
// comparison: the reference's own order at the cut is unspecified.
class RoutingOracle {
public:
    bool load(const std::string & dir, std::string & why);
    bool loaded() const { return loaded_; }
    // `ids` is [n_layers][N_EXPERT_USED] for the layer span starting at il0.
    void observe(int pos, int il0, const std::vector<int> & ids);
    // `valid_pos_end` is the first position whose INPUT TOKEN differs from the
    // reference's. Beyond it the two engines are decoding different text, so a
    // routing difference says nothing about the router -- the headline number
    // is the one over positions below it.
    void report(FILE * out, int il0, int il1, int valid_pos_end) const;

private:
    bool loaded_ = false;
    std::map<std::pair<int,int>, std::vector<int>> ref_;      // (il, pos) -> ids
    std::map<std::pair<int,int>, double>           overlap_;  // (il, pos) -> |A n B| / k
};

// Writes this engine's taps in exactly the format above, so one run can be
// the oracle of another: `--dump A` then `--oracle A` on a second run gives a
// GPU-versus-CPU check that needs no llama.cpp, and a jitter arm that answers
// how far apart two equally correct summation orders of this architecture
// land. Returns false and says why on any I/O failure.
bool dump_taps(const Recorder & rec, const std::string & dir, std::string & why);

} // namespace fk
