// Generic utility nodes, ports of mediapipe/calculators/core/*:
// gates (AllowIf/DisallowIf), CollectionHasMinSize, ClipVectorSize,
// ConcatenateVector and per-item filtering by flags.
#pragma once

#include <vector>

#include "mlxpipe/framework/graph.h"

namespace mlxpipe {

// GateCalculator: passes `input` when `allow` is true (or, with
// `disallow_when_true`, when it is false).
template <typename T>
class GateNode : public Node {
public:
    explicit GateNode(bool disallow_when_true = false) : Node("Gate"), disallow_(disallow_when_true) {}

    Input<T> input{this, "input"};
    Input<bool> condition{this, "condition"};
    Output<T> output{this, "output"};

    void process() override {
        if (input.has() && condition.get() != disallow_) output.send(input.get());
    }

private:
    bool disallow_;
};

// CollectionHasMinSizeCalculator.
template <typename T>
class HasMinSizeNode : public Node {
public:
    explicit HasMinSizeNode(size_t min_size) : Node("CollectionHasMinSize"), min_size_(min_size) {}

    Input<std::vector<T>> items{this, "items"};
    Output<bool> result{this, "result"};

    void process() override { result.send(items.get().size() >= min_size_); }

private:
    size_t min_size_;
};

// ClipVectorSizeCalculator.
template <typename T>
class ClipVectorSizeNode : public Node {
public:
    explicit ClipVectorSizeNode(size_t max_size) : Node("ClipVectorSize"), max_size_(max_size) {}

    Input<std::vector<T>> input{this, "input"};
    Output<std::vector<T>> output{this, "output"};

    void process() override {
        std::vector<T> items = input.get();
        if (items.size() > max_size_) items.resize(max_size_);
        output.send(std::move(items));
    }

private:
    size_t max_size_;
};

// ConcatenateVectorCalculator for two inputs; empty inputs are skipped.
template <typename T>
class ConcatenateVectorNode : public Node {
public:
    ConcatenateVectorNode() : Node("ConcatenateVector") {}

    Input<std::vector<T>> first{this, "first", /*is_optional=*/true};
    Input<std::vector<T>> second{this, "second", /*is_optional=*/true};
    Output<std::vector<T>> output{this, "output"};

    void process() override {
        std::vector<T> items;
        if (first.has()) items = first.get();
        if (second.has()) items.insert(items.end(), second.get().begin(), second.get().end());
        output.send(std::move(items));
    }
};

// Keeps the items whose flag is true: the batched form of gating each item
// of a MediaPipe Begin/EndLoop with AllowIf.
template <typename T>
class FilterByFlagsNode : public Node {
public:
    FilterByFlagsNode() : Node("FilterByFlags") {}

    Input<std::vector<T>> items{this, "items"};
    Input<std::vector<bool>> flags{this, "flags"};
    Output<std::vector<T>> output{this, "output"};

    void process() override {
        std::vector<T> kept;
        for (size_t i = 0; i < items.get().size(); ++i) {
            if (flags.get().at(i)) kept.push_back(items.get()[i]);
        }
        output.send(std::move(kept));
    }
};

// Element-wise AND of two flag vectors; `b` may be unconnected or empty.
class AndFlagsNode : public Node {
public:
    AndFlagsNode() : Node("AndFlags") {}

    Input<std::vector<bool>> a{this, "a"};
    Input<std::vector<bool>> b{this, "b", /*is_optional=*/true};
    Output<std::vector<bool>> output{this, "output"};

    void process() override {
        std::vector<bool> flags = a.get();
        if (b.has()) {
            for (size_t i = 0; i < flags.size(); ++i) flags[i] = flags[i] && b.get().at(i);
        }
        output.send(std::move(flags));
    }
};

}  // namespace mlxpipe
