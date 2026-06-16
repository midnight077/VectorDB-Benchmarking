#pragma once

namespace hnsw {

// Squared L2 distance. Smaller = closer.
struct ScalarL2 {
    static float dist(const float* a, const float* b, int d) {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
};

// Negative inner product, so smaller = closer (matches HNSW's L2 convention).
// For cosine, vectors are L2-normalized centrally by the harness, so IP == cosine.
struct ScalarIP {
    static float dist(const float* a, const float* b, int d) {
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum += a[i] * b[i];
        }
        return -sum;
    }
};

}  // namespace hnsw
