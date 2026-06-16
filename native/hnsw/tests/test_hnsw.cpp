// Standalone correctness tests for the templated HNSW core (Phase A: scalar,
// serial build, single-thread search). No external test framework: each check
// prints PASS/FAIL and main() returns nonzero if anything failed.
//
// Build:
//   cmake -S native/hnsw -B native/hnsw/build-test -DBUILD_TESTS=ON
//   cmake --build native/hnsw/build-test
//   ./native/hnsw/build-test/test_hnsw

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "hnsw/distances_scalar.hpp"
#include "hnsw/hnsw_index.hpp"

using namespace hnsw;

namespace {

int g_passed = 0;
int g_failed = 0;

void check(bool cond, const char* msg) {
    if (cond) {
        std::printf("  PASS: %s\n", msg);
        ++g_passed;
    } else {
        std::printf("  FAIL: %s\n", msg);
        ++g_failed;
    }
}

std::vector<float> random_vectors(int n, int dim, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(static_cast<size_t>(n) * dim);
    for (auto& x : v) x = dist(gen);
    return v;
}

void l2_normalize(std::vector<float>& data, int n, int dim) {
    for (int i = 0; i < n; ++i) {
        float* row = &data[static_cast<size_t>(i) * dim];
        float norm = 0.0f;
        for (int d = 0; d < dim; ++d) norm += row[d] * row[d];
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (int d = 0; d < dim; ++d) row[d] /= norm;
        }
    }
}

// Exact brute-force top-k for ground truth.
template <class Dist>
std::vector<std::vector<int>> brute_force_knn(const std::vector<float>& data, int n, int dim,
                                               const std::vector<float>& queries, int nq, int k) {
    std::vector<std::vector<int>> result(nq);
    for (int qi = 0; qi < nq; ++qi) {
        const float* q = &queries[static_cast<size_t>(qi) * dim];
        std::vector<std::pair<float, int>> dists(n);
        for (int i = 0; i < n; ++i) {
            const float* p = &data[static_cast<size_t>(i) * dim];
            dists[i] = {Dist::dist(q, p, dim), i};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        result[qi].resize(k);
        for (int j = 0; j < k; ++j) result[qi][j] = dists[j].second;
    }
    return result;
}

double recall_at_k(const std::vector<std::vector<int>>& retrieved,
                    const std::vector<std::vector<int>>& gt, int k) {
    double total = 0.0;
    for (size_t i = 0; i < retrieved.size(); ++i) {
        std::set<int> gt_set(gt[i].begin(), gt[i].begin() + k);
        int hits = 0;
        for (int j = 0; j < k; ++j) {
            if (gt_set.count(retrieved[i][j])) ++hits;
        }
        total += static_cast<double>(hits) / k;
    }
    return total / static_cast<double>(retrieved.size());
}

template <class Dist>
double recall_test(const char* label, const std::vector<float>& data, int n, int dim,
                    const std::vector<float>& queries, int nq, int k, int M, int ef_construction,
                    int ef_search) {
    HnswIndex<Dist> index(dim, M, ef_construction, /*seed=*/42);

    auto t0 = std::chrono::high_resolution_clock::now();
    index.add_points(data.data(), n);
    auto t1 = std::chrono::high_resolution_clock::now();
    double build_s = std::chrono::duration<double>(t1 - t0).count();

    index.set_ef(ef_search);

    auto gt = brute_force_knn<Dist>(data, n, dim, queries, nq, k);

    std::vector<std::vector<int>> retrieved(nq, std::vector<int>(k));
    auto s0 = std::chrono::high_resolution_clock::now();
    for (int qi = 0; qi < nq; ++qi) {
        index.search(&queries[static_cast<size_t>(qi) * dim], k, retrieved[qi].data());
    }
    auto s1 = std::chrono::high_resolution_clock::now();
    double search_us = std::chrono::duration<double, std::micro>(s1 - s0).count() / nq;

    double recall = recall_at_k(retrieved, gt, k);
    std::printf("[%s] N=%d dim=%d M=%d efC=%d efS=%d build=%.3fs avg_search=%.1fus recall@%d=%.4f\n",
                 label, n, dim, M, ef_construction, ef_search, build_s, search_us, k, recall);
    return recall;
}

}  // namespace

int main() {
    const int dim = 64;
    const int n = 10000;
    const int nq = 200;
    const int k = 10;
    const int M = 16;
    const int ef_construction = 200;
    const int ef_search = 100;

    std::printf("== Recall vs brute-force ==\n");

    auto data_l2 = random_vectors(n, dim, /*seed=*/1);
    auto queries_l2 = random_vectors(nq, dim, /*seed=*/2);
    double recall_l2 = recall_test<ScalarL2>("ScalarL2", data_l2, n, dim, queries_l2, nq, k, M,
                                              ef_construction, ef_search);
    check(recall_l2 > 0.95, "ScalarL2 recall@10 > 0.95");

    auto data_ip = random_vectors(n, dim, /*seed=*/3);
    auto queries_ip = random_vectors(nq, dim, /*seed=*/4);
    l2_normalize(data_ip, n, dim);
    l2_normalize(queries_ip, nq, dim);
    double recall_ip = recall_test<ScalarIP>("ScalarIP", data_ip, n, dim, queries_ip, nq, k, M,
                                              ef_construction, ef_search);
    check(recall_ip > 0.95, "ScalarIP recall@10 > 0.95");

    std::printf("\n== Determinism (serial build, fixed seed) ==\n");
    {
        const int small_n = 2000;
        auto data = random_vectors(small_n, dim, /*seed=*/5);
        auto queries = random_vectors(20, dim, /*seed=*/6);

        HnswIndex<ScalarL2> idx_a(dim, M, ef_construction, /*seed=*/42);
        idx_a.add_points(data.data(), small_n);
        idx_a.set_ef(ef_search);

        HnswIndex<ScalarL2> idx_b(dim, M, ef_construction, /*seed=*/42);
        idx_b.add_points(data.data(), small_n);
        idx_b.set_ef(ef_search);

        bool identical = true;
        for (int qi = 0; qi < 20; ++qi) {
            std::vector<int> ra(k), rb(k);
            idx_a.search(&queries[static_cast<size_t>(qi) * dim], k, ra.data());
            idx_b.search(&queries[static_cast<size_t>(qi) * dim], k, rb.data());
            if (ra != rb) {
                identical = false;
                break;
            }
        }
        check(identical, "two serial builds with the same seed give identical search results");
    }

    std::printf("\n== Save / load roundtrip ==\n");
    {
        const int small_n = 2000;
        auto data = random_vectors(small_n, dim, /*seed=*/7);
        auto queries = random_vectors(20, dim, /*seed=*/8);

        HnswIndex<ScalarL2> idx(dim, M, ef_construction, /*seed=*/42);
        idx.add_points(data.data(), small_n);
        idx.set_ef(ef_search);

        const std::string path = "/tmp/vdbhnsw_test_save.bin";
        idx.save(path);

        HnswIndex<ScalarL2> loaded(dim, M, ef_construction, /*seed=*/0);
        loaded.load(path);
        loaded.set_ef(ef_search);

        check(loaded.size() == idx.size(), "loaded index has the same number of points");

        bool identical = true;
        for (int qi = 0; qi < 20; ++qi) {
            std::vector<int> ra(k), rb(k);
            idx.search(&queries[static_cast<size_t>(qi) * dim], k, ra.data());
            loaded.search(&queries[static_cast<size_t>(qi) * dim], k, rb.data());
            if (ra != rb) {
                identical = false;
                break;
            }
        }
        check(identical, "loaded index gives identical search results to the original");
    }

    std::printf("\n%d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
