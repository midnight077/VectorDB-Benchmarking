#pragma once

#include <cstdint>
#include <vector>

namespace hnsw {

// Epoch-counter visited set: avoids reallocating/clearing a bitset per query.
// Each id's "visited" state is encoded by whether tags_[id] == current tag.
class VisitedPool {
public:
    explicit VisitedPool(size_t size = 0) : tags_(size, 0), cur_tag_(0) {}

    void resize(size_t size) {
        tags_.assign(size, 0);
        cur_tag_ = 0;
    }

    size_t capacity_bytes() const { return tags_.capacity() * sizeof(uint32_t); }

    // A lightweight view bound to the current epoch tag.
    class VisitedList {
    public:
        VisitedList(std::vector<uint32_t>& tags, uint32_t tag) : tags_(tags), tag_(tag) {}

        bool is_visited(uint32_t id) const { return tags_[id] == tag_; }
        void set_visited(uint32_t id) { tags_[id] = tag_; }

    private:
        std::vector<uint32_t>& tags_;
        uint32_t tag_;
    };

    // Bump the epoch and hand back a view for a fresh query. On overflow,
    // reset all tags to 0 and restart numbering at 1.
    VisitedList get() {
        ++cur_tag_;
        if (cur_tag_ == 0) {
            std::fill(tags_.begin(), tags_.end(), 0);
            cur_tag_ = 1;
        }
        return VisitedList(tags_, cur_tag_);
    }

private:
    std::vector<uint32_t> tags_;
    uint32_t cur_tag_;
};

}  // namespace hnsw
