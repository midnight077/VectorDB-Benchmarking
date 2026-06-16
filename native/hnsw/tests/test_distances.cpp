// Phase B: SIMD-vs-scalar distance kernel unit test (no OpenMP / build-path
// changes here -- see CUSTOM_HNSW_PLAN.md Phase B).
//
// Compares SimdL2/SimdIP (include/hnsw/distances_simd.hpp) against
// ScalarL2/ScalarIP (include/hnsw/distances_scalar.hpp) on random vectors
// across a range of dimensions, including dim=100 (12*8 + 4) to exercise the
// load_partial/store_partial remainder path that a multiple-of-8 dim like
// 128/784/1024 would never touch.
//
// Build:
//   cmake -S native/hnsw -B native/hnsw/build-test -DBUILD_TESTS=ON
//   cmake --build native/hnsw/build-test
//   ./native/hnsw/build-test/test_distances

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "hnsw/distances_scalar.hpp"
#include "hnsw/distances_simd.hpp"

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

// Max over `pairs` of |simd - scalar| / max(|scalar|, |simd|, 1.0).
//
// The floor of 1.0 matters for IP/cosine: with L2-normalized vectors the
// true inner product concentrates near 0 in high dimensions (two random unit
// vectors in R^1024 are nearly orthogonal), so a plain relative error
// |simd-scalar|/|scalar| blows up even though the *absolute* difference is a
// tiny float-rounding artifact (~1e-4). Cosine values are bounded to [-1,1],
// so the floor turns this into an absolute-error check in that regime while
// still giving a true relative check for L2, whose squared distances are
// bounded away from zero for distinct random vectors.
template <class Scalar, class Simd>
float max_relative_diff(const std::vector<float>& data, int n, int dim,
                         const std::vector<float>& queries, int nq) {
    float max_rel = 0.0f;
    for (int qi = 0; qi < nq; ++qi) {
        const float* q = &queries[static_cast<size_t>(qi) * dim];
        for (int i = 0; i < n; ++i) {
            const float* p = &data[static_cast<size_t>(i) * dim];
            float ds = Scalar::dist(q, p, dim);
            float dv = Simd::dist(q, p, dim);
            float denom = std::max({std::fabs(ds), std::fabs(dv), 1.0f});
            float rel = std::fabs(dv - ds) / denom;
            max_rel = std::max(max_rel, rel);
        }
    }
    return max_rel;
}

// Runs the L2 and IP comparisons for one dimension, printing the max
// relative diff for each and checking it against `tol`.
void run_dim(int dim, float tol) {
    const int n = 200;
    const int nq = 50;

    std::printf("-- dim=%d --\n", dim);

    auto data_l2 = random_vectors(n, dim, /*seed=*/100 + static_cast<uint32_t>(dim));
    auto queries_l2 = random_vectors(nq, dim, /*seed=*/200 + static_cast<uint32_t>(dim));
    float rel_l2 = max_relative_diff<ScalarL2, SimdL2>(data_l2, n, dim, queries_l2, nq);
    std::printf("  L2: max relative diff = %.3e\n", rel_l2);
    char msg_l2[128];
    std::snprintf(msg_l2, sizeof(msg_l2), "dim=%d SimdL2 vs ScalarL2 relative diff < %.0e", dim,
                  static_cast<double>(tol));
    check(rel_l2 < tol, msg_l2);

    // IP / cosine: harness L2-normalizes vectors centrally before handing
    // them to the index, so exercise IP on normalized vectors too.
    auto data_ip = random_vectors(n, dim, /*seed=*/300 + static_cast<uint32_t>(dim));
    auto queries_ip = random_vectors(nq, dim, /*seed=*/400 + static_cast<uint32_t>(dim));
    l2_normalize(data_ip, n, dim);
    l2_normalize(queries_ip, nq, dim);
    float rel_ip = max_relative_diff<ScalarIP, SimdIP>(data_ip, n, dim, queries_ip, nq);
    std::printf("  IP: max relative diff = %.3e\n", rel_ip);
    char msg_ip[128];
    std::snprintf(msg_ip, sizeof(msg_ip), "dim=%d SimdIP vs ScalarIP relative diff < %.0e", dim,
                  static_cast<double>(tol));
    check(rel_ip < tol, msg_ip);
}

}  // namespace

int main() {
    std::printf("SimdF::size() = %d (vector width in floats)\n\n", SimdF::size());

    const float tol = 1e-3f;

    // 1, 7: smaller than the vector width, pure remainder path.
    // 8, 16: exactly one/two full vector widths (Vec8f), no remainder.
    // 100: 12*8 + 4 -- the canonical "not a multiple of 8 or 16" case
    //      (glove-100 dimensionality). Most important case.
    // 128, 256: SIFT-like, multiples of both 8 and 16.
    // 784: fashion-mnist, 784 = 98*8 (multiple of 8, not of 16).
    // 1024: bge-m3, multiple of both.
    const int dims[] = {1, 7, 8, 16, 100, 128, 256, 784, 1024};

    for (int dim : dims) {
        run_dim(dim, tol);
    }

    std::printf("\n%d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
