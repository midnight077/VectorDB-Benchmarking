#pragma once

#include "vectorclass.h"

namespace hnsw {

// Vector width selected at compile time: Vec16f (AVX-512) if the target
// supports AVX512F, otherwise Vec8f (AVX2/FMA). Picked via -march=native.
#if defined(__AVX512F__)
using SimdF = Vec16f;
#else
using SimdF = Vec8f;
#endif

// Squared L2 distance via VCL SIMD. Smaller = closer. Same convention as
// ScalarL2 (distances_scalar.hpp); reduction order differs so results match
// only to within float rounding.
struct SimdL2 {
    static float dist(const float* a, const float* b, int d) {
        constexpr int W = SimdF::size();
        SimdF acc(0.0f);
        int i = 0;
        for (; i + W <= d; i += W) {
            SimdF va = SimdF().load(a + i);
            SimdF vb = SimdF().load(b + i);
            SimdF diff = va - vb;
            acc = mul_add(diff, diff, acc);
        }
        float sum = horizontal_add(acc);

        const int rem = d - i;
        if (rem > 0) {
            // load_partial zero-fills the remaining lanes, so the squared
            // difference in those lanes is 0 and doesn't perturb the sum.
            SimdF va = SimdF().load_partial(rem, a + i);
            SimdF vb = SimdF().load_partial(rem, b + i);
            SimdF diff = va - vb;
            sum += horizontal_add(diff * diff);
        }
        return sum;
    }
};

// Negative inner product via VCL SIMD, so smaller = closer (matches HNSW's
// L2 convention). For cosine, vectors are L2-normalized centrally by the
// harness, so IP == cosine.
struct SimdIP {
    static float dist(const float* a, const float* b, int d) {
        constexpr int W = SimdF::size();
        SimdF acc(0.0f);
        int i = 0;
        for (; i + W <= d; i += W) {
            SimdF va = SimdF().load(a + i);
            SimdF vb = SimdF().load(b + i);
            acc = mul_add(va, vb, acc);
        }
        float sum = horizontal_add(acc);

        const int rem = d - i;
        if (rem > 0) {
            SimdF va = SimdF().load_partial(rem, a + i);
            SimdF vb = SimdF().load_partial(rem, b + i);
            sum += horizontal_add(va * vb);
        }
        return -sum;
    }
};

}  // namespace hnsw
