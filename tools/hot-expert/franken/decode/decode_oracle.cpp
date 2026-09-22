// tools/hot-expert/franken/decode/decode_oracle.cpp -- see decode_oracle.h.

#include "decode_oracle.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>

namespace fk {
namespace {

bool file_exists(const std::string & p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// The index's key IS the file stem. The two spellings the brief and the tool
// could disagree on ("<name>-<il>.f32" vs "<name>.f32" for il < 0) are both
// tried, so a dumper change shows up as MISSING rather than as a silent pass.
std::vector<std::string> candidate_paths(const std::string & dir, const std::string & key, int il) {
    std::vector<std::string> v;
    v.push_back(dir + "/" + key + ".f32");
    if (il >= 0) v.push_back(dir + "/" + key + "-" + std::to_string(il) + ".f32");
    return v;
}

// Points the dump carries but that a DECODE run at a DIFFERENT cache depth
// cannot be held to. Neither is skipped silently: each prints its reason and
// is counted as INCOMPARABLE, never as a pass.
//
//  indexer_k_pooled / indexer_k  the file is the last BLOCK's column, and the
//      last block is llama.cpp's padded n_kv/ratio (block 63 for a 6-token
//      prompt in a 256-cell cache) -- a block this engine's exactly-sized
//      cache does not have. The selection these keys drive IS checked:
//      indexer_score compares every block this engine does have.
//  state_predelta  llama.cpp evaluated the six tokens as ONE ubatch, so the
//      state it read into the graph is the pre-batch (zero) state, while a
//      decode loop reads the state after five tokens. The recurrence itself
//      is checked by attn_output, which is its output for the same token.
const char * incomparable_reason(const std::string & name) {
    if (name == "indexer_k_pooled" || name == "indexer_k")
        return "the dump's column is the last block of llama.cpp's PADDED cache";
    if (name == "state_predelta")
        return "llama.cpp ran the prompt as one ubatch, so its pre-state is the zero state";
    return nullptr;
}

} // namespace

bool Oracle::load(const std::string & dir, std::string & why) {
    dir_ = dir;
    const std::string index = dir + "/index.txt";
    FILE * f = std::fopen(index.c_str(), "r");
    if (!f) {
        why = "no " + index + " (deliverable A has not produced the dump yet)";
        return false;
    }
    char line[512];
    while (std::fgets(line, sizeof(line), f)) {
        std::istringstream is(line);
        OracleEntry e;
        if (!(is >> e.key >> e.il)) continue;
        is >> e.ne0 >> e.ne1 >> e.ne2 >> e.type;
        for (const auto & p : candidate_paths(dir, e.key, e.il)) {
            if (file_exists(p)) { e.path = p; break; }
        }
        if (e.path.empty()) continue;
        entries_[e.key] = e;
    }
    std::fclose(f);
    if (entries_.empty()) {
        why = index + " listed no readable .f32 file";
        return false;
    }
    loaded_ = true;
    return true;
}

bool Oracle::read_file(const OracleEntry & e, std::vector<float> & out) const {
    FILE * f = std::fopen(e.path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize((size_t) n / sizeof(float));
    const size_t got = std::fread(out.data(), sizeof(float), out.size(), f);
    std::fclose(f);
    out.resize(got);
    return !out.empty();
}

bool Oracle::compare(const Recorder & rec, const std::vector<int> & pass_layers,
                     double min_cos, FILE * out) {
    bool all_ok = true;
    int  n_cmp = 0, n_missing = 0, n_incomp = 0;
    std::set<std::string> seen;

    for (const auto & kv : rec.all()) {
        const TapValue & t = kv.second;
        auto it = entries_.find(t.key);
        if (it == entries_.end()) {
            // A tap this engine computes that the dump does not carry. Some
            // are expected: ggml_set_name overwrites, so `kqv_out` and the
            // second `hc_combine` never reach the dump under those names.
            // Never counted as a pass.
            std::fprintf(out, "oracle %s MISSING (no dump file; not counted as a pass)\n",
                         t.key.c_str());
            ++n_missing;
            continue;
        }
        seen.insert(t.key);

        if (const char * why = incomparable_reason(t.name)) {
            std::fprintf(out, "oracle %s INCOMPARABLE (%s)\n", t.key.c_str(), why);
            ++n_incomp;
            continue;
        }

        std::vector<float> ref;
        if (!read_file(it->second, ref)) {
            std::fprintf(out, "oracle %s UNREADABLE %s\n", t.key.c_str(), it->second.path.c_str());
            all_ok = false;
            continue;
        }

        // `indexer_top_k` is a SET of cell indices, and the two sides do not
        // have the same cache depth: llama.cpp pads n_kv (256 cells for a
        // 6-token prompt), so its budget of min(n_kv, top_k+r-1) cells is
        // filled out with cells this engine's exact-length cache does not
        // have. The test is therefore CONTAINMENT -- every cell this engine
        // selected must be one the reference selected -- plus both sizes
        // printed, which is an exact set match whenever the depths agree.
        // The reference's own ggml_top_k and the radix select both break ties
        // at the cut in an unspecified order, so nothing stricter is sound.
        if (t.name == "indexer_top_k") {
            std::set<int> a, b;
            for (float v : t.data) a.insert((int) v);
            for (float v : ref)    b.insert((int) v);
            std::vector<int> inter;
            std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(inter));
            const double frac = a.empty() ? 0.0 : (double) inter.size() / (double) a.size();
            const bool ok = !a.empty() && !b.empty() && frac >= 1.0;
            std::fprintf(out, "oracle %s contained=%.6f mine=%zu ref=%zu%s\n",
                         t.key.c_str(), frac, a.size(), b.size(), ok ? "" : " FAIL");
            if (!ok) all_ok = false;
            ++n_cmp;
            continue;
        }

        const size_t n = std::min(t.data.size(), ref.size());
        if (n == 0) {
            std::fprintf(out, "oracle %s REFUSED (one side is empty)\n", t.key.c_str());
            all_ok = false;
            continue;
        }
        double dot = 0.0, na = 0.0, nb = 0.0, maxabs = 0.0, l1 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double a = t.data[i], b = ref[i];
            dot += a * b; na += a * a; nb += b * b;
            maxabs = std::max(maxabs, std::fabs(a - b));
            l1 += std::fabs(b);
        }
        const bool degenerate = (na <= 0.0 || nb <= 0.0);
        const double cos = degenerate ? 0.0 : dot / (std::sqrt(na) * std::sqrt(nb));
        const bool ok = !degenerate && cos >= min_cos;
        std::fprintf(out, "oracle %s cos=%.6f maxabs=%.6g ref_l1=%.6g n=%zu%s%s%s\n",
                     t.key.c_str(), cos, maxabs, l1, n,
                     t.data.size() != ref.size() ? " (compared the common prefix)" : "",
                     degenerate ? " ZERO-NORM" : "",
                     ok ? "" : " FAIL");
        if (!ok) all_ok = false;
        ++n_cmp;
    }

    for (const auto & kv : entries_) {
        if (!seen.count(kv.first))
            std::fprintf(out, "oracle %s UNCHECKED (dumped, not computed by this range)\n",
                         kv.first.c_str());
    }

    // The brief's explicit bar: the layer outputs must pass, not just the
    // easy pointwise taps.
    for (int il : pass_layers) {
        for (const char * nm : {"l_last", "hc_combine"}) {
            const std::string key = std::string(nm) + "-" + std::to_string(il);
            const TapValue * t = rec.get(key);
            const bool have_ref = entries_.count(key) != 0;
            if (!t || !have_ref) {
                std::fprintf(out, "oracle %s REQUIRED but %s\n", key.c_str(),
                             t ? "absent from the dump" : "not computed");
                all_ok = false;
            }
        }
    }

    std::fprintf(out, "oracle summary: compared=%d missing=%d incomparable=%d dir=%s\n",
                 n_cmp, n_missing, n_incomp, dir_.c_str());
    return all_ok && n_cmp > 0;
}

bool dump_taps(const Recorder & rec, const std::string & dir, std::string & why) {
    ::mkdir(dir.c_str(), 0755);   // fine if it already exists
    const std::string index = dir + "/index.txt";
    FILE * ix = std::fopen(index.c_str(), "w");
    if (!ix) { why = "cannot write " + index; return false; }
    for (const auto & kv : rec.all()) {
        const TapValue & t = kv.second;
        const std::string path = dir + "/" + t.key + ".f32";
        FILE * f = std::fopen(path.c_str(), "wb");
        if (!f) { why = "cannot write " + path; std::fclose(ix); return false; }
        std::fwrite(t.data.data(), sizeof(float), t.data.size(), f);
        std::fclose(f);
        // ne1/ne2 are 1: the payload IS the slice, so a reader that applies
        // the dump's "last slice of the slowest non-unit axis" rule to it
        // gets the whole thing back.
        std::fprintf(ix, "%s %d %zu 1 1 f32\n", t.key.c_str(), t.il, t.data.size());
    }
    std::fclose(ix);
    return true;
}

} // namespace fk
