#include "mlxpipe/calculators/association.h"

#include <list>

#include "mlxpipe/calculators/hand.h"

namespace mlxpipe {

void AssociationNormRectNode::process() {
    struct Entry {
        NormalizedRect rect;
        int base_index;  // -1 for rects from `rects`
    };
    std::list<Entry> result;
    auto add = [&](NormalizedRect rect, int base_index) {
        for (auto it = result.begin(); it != result.end();) {
            if (RectIou(rect, it->rect) > threshold_) {
                if (it->rect.rect_id != 0) rect.rect_id = it->rect.rect_id;
                it = result.erase(it);
            } else {
                ++it;
            }
        }
        if (rect.rect_id == 0) rect.rect_id = next_rect_id_++;
        result.push_back({rect, base_index});
    };

    const size_t num_base = base_rects.has() ? base_rects.get().size() : 0;
    for (size_t i = 0; i < num_base; ++i) add(base_rects.get()[i], static_cast<int>(i));
    if (rects.has()) {
        for (const auto& rect : rects.get()) add(rect, -1);
    }

    std::vector<NormalizedRect> all, added;
    std::vector<bool> kept(num_base, false);
    for (const auto& entry : result) {
        if (max_rects_ > 0 && all.size() >= max_rects_) break;
        all.push_back(entry.rect);
        if (entry.base_index >= 0) {
            kept[entry.base_index] = true;
        } else {
            added.push_back(entry.rect);
        }
    }
    output.send(std::move(all));
    base_kept.send(std::move(kept));
    new_rects.send(std::move(added));
}

}  // namespace mlxpipe
