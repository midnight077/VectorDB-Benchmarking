#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "hnsw/visited_pool.hpp"

namespace hnsw {

// Templated HNSW core shared by all variants (scalar/SIMD, serial/parallel build).
// `Dist::dist(a, b, d)` must return a "smaller = closer" distance (squared L2 for
// L2, negative inner product for IP/cosine).
//
// Phase A: serial build, single-thread search. `add_points`'s `parallel`/
// `num_threads` arguments are accepted (for ABI stability with later phases) but
// ignored here.
template <class Dist>
class HnswIndex {
public:
    HnswIndex(int dim, int M, int ef_construction, uint64_t seed = 42)
        : dim_(dim),
          M_(M),
          M_max0_(2 * M),
          ef_construction_(ef_construction),
          ef_search_(10),
          mL_(1.0 / std::log(static_cast<double>(std::max(M, 2)))),
          entry_point_(0),
          max_level_(-1),
          rng_(seed) {}

    // Append n points (row-major, n x dim_) and build their graph connections.
    void add_points(const float* data, int n, bool parallel = false, int num_threads = 1) {
        (void)parallel;
        (void)num_threads;

        size_t start_id = data_.size() / static_cast<size_t>(dim_);
        data_.resize((start_id + static_cast<size_t>(n)) * static_cast<size_t>(dim_));
        std::memcpy(data_.data() + start_id * static_cast<size_t>(dim_), data,
                    static_cast<size_t>(n) * static_cast<size_t>(dim_) * sizeof(float));

        levels_.resize(start_id + static_cast<size_t>(n));
        links_.resize(start_id + static_cast<size_t>(n));
        visited_pool_.resize(start_id + static_cast<size_t>(n));

        for (int i = 0; i < n; ++i) {
            add_point(static_cast<uint32_t>(start_id + static_cast<size_t>(i)));
        }
    }

    // Search-time parameter; takes effect on the next search(), no rebuild.
    void set_ef(int ef_search) { ef_search_ = ef_search; }

    // Single query, single thread. Writes k ids (dataset row indices) into out_ids,
    // padding with -1 if the index holds fewer than k points.
    void search(const float* query, int k, int* out_ids) const {
        for (int i = 0; i < k; ++i) out_ids[i] = -1;
        if (size() == 0) return;

        uint32_t cur_ep = entry_point_;
        float cur_dist = Dist::dist(query, get_data(cur_ep), dim_);
        for (int level = max_level_; level > 0; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (uint32_t e : links_[cur_ep][level]) {
                    float d = Dist::dist(query, get_data(e), dim_);
                    if (d < cur_dist) {
                        cur_dist = d;
                        cur_ep = e;
                        changed = true;
                    }
                }
            }
        }

        int ef = std::max(ef_search_, k);
        auto visited = visited_pool_.get();
        MaxHeap top_candidates = search_layer(query, cur_ep, ef, 0, visited);

        while (top_candidates.size() > static_cast<size_t>(k)) {
            top_candidates.pop();
        }

        std::vector<Candidate> result;
        result.reserve(top_candidates.size());
        while (!top_candidates.empty()) {
            result.push_back(top_candidates.top());
            top_candidates.pop();
        }
        std::sort(result.begin(), result.end());

        for (size_t i = 0; i < result.size(); ++i) {
            out_ids[i] = static_cast<int>(result[i].second);
        }
    }

    size_t size() const { return levels_.size(); }
    int dim() const { return dim_; }

    // Rough resident-size estimate: data + per-level neighbor lists + bookkeeping.
    size_t index_memory_bytes() const {
        size_t total = sizeof(*this);
        total += data_.capacity() * sizeof(float);
        total += levels_.capacity() * sizeof(int);
        total += links_.capacity() * sizeof(std::vector<std::vector<uint32_t>>);
        for (const auto& per_level : links_) {
            total += per_level.capacity() * sizeof(std::vector<uint32_t>);
            for (const auto& neighbors : per_level) {
                total += neighbors.capacity() * sizeof(uint32_t);
            }
        }
        total += visited_pool_.capacity_bytes();
        return total;
    }

    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("hnsw: failed to open file for writing: " + path);

        const uint32_t magic = kMagic;
        const uint32_t version = kVersion;
        const uint64_t n = size();

        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
        out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        out.write(reinterpret_cast<const char*>(&M_), sizeof(M_));
        out.write(reinterpret_cast<const char*>(&M_max0_), sizeof(M_max0_));
        out.write(reinterpret_cast<const char*>(&ef_construction_), sizeof(ef_construction_));
        out.write(reinterpret_cast<const char*>(&ef_search_), sizeof(ef_search_));
        out.write(reinterpret_cast<const char*>(&mL_), sizeof(mL_));
        out.write(reinterpret_cast<const char*>(&entry_point_), sizeof(entry_point_));
        out.write(reinterpret_cast<const char*>(&max_level_), sizeof(max_level_));
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));

        out.write(reinterpret_cast<const char*>(data_.data()),
                  static_cast<std::streamsize>(data_.size() * sizeof(float)));
        out.write(reinterpret_cast<const char*>(levels_.data()),
                  static_cast<std::streamsize>(levels_.size() * sizeof(int)));

        for (uint64_t id = 0; id < n; ++id) {
            for (int level = 0; level <= levels_[id]; ++level) {
                const auto& neighbors = links_[id][static_cast<size_t>(level)];
                const uint32_t cnt = static_cast<uint32_t>(neighbors.size());
                out.write(reinterpret_cast<const char*>(&cnt), sizeof(cnt));
                if (cnt > 0) {
                    out.write(reinterpret_cast<const char*>(neighbors.data()),
                              static_cast<std::streamsize>(cnt * sizeof(uint32_t)));
                }
            }
        }
        if (!out) throw std::runtime_error("hnsw: write error: " + path);
    }

    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("hnsw: failed to open file for reading: " + path);

        uint32_t magic = 0, version = 0;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        in.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (magic != kMagic || version != kVersion) {
            throw std::runtime_error("hnsw: bad file header: " + path);
        }

        uint64_t n = 0;
        in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
        in.read(reinterpret_cast<char*>(&M_), sizeof(M_));
        in.read(reinterpret_cast<char*>(&M_max0_), sizeof(M_max0_));
        in.read(reinterpret_cast<char*>(&ef_construction_), sizeof(ef_construction_));
        in.read(reinterpret_cast<char*>(&ef_search_), sizeof(ef_search_));
        in.read(reinterpret_cast<char*>(&mL_), sizeof(mL_));
        in.read(reinterpret_cast<char*>(&entry_point_), sizeof(entry_point_));
        in.read(reinterpret_cast<char*>(&max_level_), sizeof(max_level_));
        in.read(reinterpret_cast<char*>(&n), sizeof(n));

        data_.resize(n * static_cast<uint64_t>(dim_));
        in.read(reinterpret_cast<char*>(data_.data()),
                static_cast<std::streamsize>(data_.size() * sizeof(float)));

        levels_.resize(n);
        in.read(reinterpret_cast<char*>(levels_.data()),
                static_cast<std::streamsize>(levels_.size() * sizeof(int)));

        links_.assign(n, {});
        for (uint64_t id = 0; id < n; ++id) {
            links_[id].resize(static_cast<size_t>(levels_[id] + 1));
            for (int level = 0; level <= levels_[id]; ++level) {
                uint32_t cnt = 0;
                in.read(reinterpret_cast<char*>(&cnt), sizeof(cnt));
                auto& neighbors = links_[id][static_cast<size_t>(level)];
                neighbors.resize(cnt);
                if (cnt > 0) {
                    in.read(reinterpret_cast<char*>(neighbors.data()),
                            static_cast<std::streamsize>(cnt * sizeof(uint32_t)));
                }
            }
        }

        visited_pool_.resize(n);

        if (!in) throw std::runtime_error("hnsw: read error: " + path);
    }

private:
    static constexpr uint32_t kMagic = 0x57534E48;  // "HNSW"
    static constexpr uint32_t kVersion = 1;

    using Candidate = std::pair<float, uint32_t>;
    // Max-heap by distance: top() = furthest.
    using MaxHeap = std::priority_queue<Candidate>;
    // Min-heap by distance: top() = nearest.
    using MinHeap = std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>>;

    const float* get_data(uint32_t id) const {
        return data_.data() + static_cast<size_t>(id) * static_cast<size_t>(dim_);
    }

    int random_level() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = -std::log(dist(rng_)) * mL_;
        return static_cast<int>(r);
    }

    // Beam search at a single layer. Returns up to `ef` nearest-to-`point` ids
    // visited from `entry`, as a max-heap (top = furthest of the kept set).
    MaxHeap search_layer(const float* point, uint32_t entry, int ef, int level,
                          VisitedPool::VisitedList& visited) const {
        MaxHeap top_candidates;
        MinHeap candidate_set;

        float d = Dist::dist(point, get_data(entry), dim_);
        top_candidates.emplace(d, entry);
        candidate_set.emplace(d, entry);
        visited.set_visited(entry);

        while (!candidate_set.empty()) {
            Candidate current = candidate_set.top();
            if (current.first > top_candidates.top().first &&
                top_candidates.size() >= static_cast<size_t>(ef)) {
                break;
            }
            candidate_set.pop();

            for (uint32_t e : links_[current.second][static_cast<size_t>(level)]) {
                if (!visited.is_visited(e)) {
                    visited.set_visited(e);
                    float dist_e = Dist::dist(point, get_data(e), dim_);
                    if (top_candidates.size() < static_cast<size_t>(ef) ||
                        dist_e < top_candidates.top().first) {
                        candidate_set.emplace(dist_e, e);
                        top_candidates.emplace(dist_e, e);
                        if (top_candidates.size() > static_cast<size_t>(ef)) {
                            top_candidates.pop();
                        }
                    }
                }
            }
        }
        return top_candidates;
    }

    // Robert-Malkov-Yashunin heuristic: prefer candidates not dominated by an
    // already-selected, closer neighbor ("diverse" neighbor set). If there are
    // <= M candidates, keep them all (matches hnswlib's getNeighborsByHeuristic2).
    std::vector<uint32_t> select_neighbors_heuristic(MaxHeap candidates, size_t M) const {
        if (candidates.size() <= M) {
            std::vector<uint32_t> result;
            result.reserve(candidates.size());
            while (!candidates.empty()) {
                result.push_back(candidates.top().second);
                candidates.pop();
            }
            return result;
        }

        std::vector<Candidate> sorted;
        sorted.reserve(candidates.size());
        while (!candidates.empty()) {
            sorted.push_back(candidates.top());
            candidates.pop();
        }
        std::reverse(sorted.begin(), sorted.end());  // ascending by distance to `point`

        std::vector<uint32_t> result;
        result.reserve(M);
        for (const auto& [dist_to_point, cand_id] : sorted) {
            if (result.size() >= M) break;
            bool good = true;
            for (uint32_t r : result) {
                if (Dist::dist(get_data(cand_id), get_data(r), dim_) < dist_to_point) {
                    good = false;
                    break;
                }
            }
            if (good) result.push_back(cand_id);
        }
        return result;
    }

    // Add `new_id` to `neighbor_id`'s adjacency at `level`, pruning with the
    // heuristic if that exceeds the level's degree cap.
    void connect_and_prune(uint32_t new_id, uint32_t neighbor_id, int level) {
        auto& neighbor_links = links_[neighbor_id][static_cast<size_t>(level)];
        for (uint32_t existing : neighbor_links) {
            if (existing == new_id) return;
        }

        const size_t max_conn = (level == 0) ? static_cast<size_t>(M_max0_) : static_cast<size_t>(M_);
        if (neighbor_links.size() < max_conn) {
            neighbor_links.push_back(new_id);
            return;
        }

        const float* neighbor_point = get_data(neighbor_id);
        MaxHeap candidates;
        candidates.emplace(Dist::dist(neighbor_point, get_data(new_id), dim_), new_id);
        for (uint32_t x : neighbor_links) {
            candidates.emplace(Dist::dist(neighbor_point, get_data(x), dim_), x);
        }
        neighbor_links = select_neighbors_heuristic(candidates, max_conn);
    }

    void add_point(uint32_t id) {
        const float* point = get_data(id);
        const int new_level = random_level();
        levels_[id] = new_level;
        links_[id].resize(static_cast<size_t>(new_level) + 1);

        if (max_level_ < 0) {
            entry_point_ = id;
            max_level_ = new_level;
            return;
        }

        uint32_t cur_ep = entry_point_;
        const int cur_max_level = max_level_;

        // Greedily descend from the top level down to new_level + 1.
        if (new_level < cur_max_level) {
            float cur_dist = Dist::dist(point, get_data(cur_ep), dim_);
            for (int level = cur_max_level; level > new_level; --level) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    for (uint32_t e : links_[cur_ep][static_cast<size_t>(level)]) {
                        float d = Dist::dist(point, get_data(e), dim_);
                        if (d < cur_dist) {
                            cur_dist = d;
                            cur_ep = e;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Beam search + connect at each level from min(new_level, cur_max_level) down to 0.
        const int top = std::min(new_level, cur_max_level);
        for (int level = top; level >= 0; --level) {
            auto visited = visited_pool_.get();
            MaxHeap candidates = search_layer(point, cur_ep, ef_construction_, level, visited);
            if (!candidates.empty()) {
                cur_ep = candidates.top().second;
            }

            const size_t max_conn = (level == 0) ? static_cast<size_t>(M_max0_) : static_cast<size_t>(M_);
            std::vector<uint32_t> selected = select_neighbors_heuristic(candidates, max_conn);
            links_[id][static_cast<size_t>(level)] = selected;

            for (uint32_t e : selected) {
                connect_and_prune(id, e, level);
            }
        }

        if (new_level > cur_max_level) {
            max_level_ = new_level;
            entry_point_ = id;
        }
    }

    int dim_;
    int M_;
    int M_max0_;
    int ef_construction_;
    int ef_search_;
    double mL_;
    uint32_t entry_point_;
    int max_level_;

    std::vector<float> data_;                          // size() * dim_, row-major
    std::vector<int> levels_;                          // per-point top level
    std::vector<std::vector<std::vector<uint32_t>>> links_;  // links_[id][level] = neighbor ids

    mutable VisitedPool visited_pool_;
    std::mt19937_64 rng_;
};

}  // namespace hnsw
