#include "mlxpipe/framework/graph.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace mlxpipe {

bool Graph::is_loopback(const Node* node) const {
    return std::any_of(loopbacks_.begin(), loopbacks_.end(), [node](const LoopbackHooks& h) { return h.node == node; });
}

// Validates connections and computes a topological order. Edges into
// loopback nodes are back edges and are ignored.
void Graph::prepare() {
    std::unordered_map<const Node*, std::vector<Node*>> dependents;
    std::unordered_map<const Node*, int> indegree;
    for (const auto& node : nodes_) indegree[node.get()] = 0;

    for (const auto& node : nodes_) {
        if (is_loopback(node.get())) continue;
        std::unordered_set<const Node*> producers;
        for (const auto* input : node->inputs_) {
            if (!input->stream()) {
                if (input->optional) continue;
                throw std::logic_error("Required input '" + input->name + "' of node '" + node->type_name() +
                                       "' is not connected");
            }
            const Node* producer = input->stream()->producer;
            if (producer && producers.insert(producer).second) {
                dependents[producer].push_back(node.get());
                ++indegree[node.get()];
            }
        }
    }

    order_.clear();
    std::vector<Node*> ready;
    for (const auto& node : nodes_) {
        if (indegree[node.get()] == 0) ready.push_back(node.get());
    }
    // Keep insertion order among ready nodes so the order is deterministic.
    size_t head = 0;
    while (head < ready.size()) {
        Node* node = ready[head++];
        order_.push_back(node);
        for (Node* dependent : dependents[node]) {
            if (--indegree[dependent] == 0) ready.push_back(dependent);
        }
    }
    if (order_.size() != nodes_.size()) {
        throw std::logic_error("Graph has a cycle; use Graph::loopback for back edges");
    }
    prepared_ = true;
}

void Graph::run(Timestamp timestamp) {
    if (!prepared_) prepare();

    for (const auto& node : nodes_) {
        for (auto* output : node->outputs_) output->stream()->clear();
    }
    for (auto& loopback : loopbacks_) loopback.emit();

    const size_t n = order_.size();
    std::vector<bool> done(n, false);
    std::unordered_set<const Node*> finished;
    for (auto& loopback : loopbacks_) finished.insert(loopback.node);

    auto inputs_resolved = [&](const Node* node) {
        for (const auto* input : node->inputs_) {
            const internal::StreamState* stream = input->stream();
            if (stream && stream->producer && !finished.count(stream->producer)) return false;
        }
        return true;
    };
    // MediaPipe's default input policy: run when every required input has a
    // value and, if the node has inputs, at least one of them does.
    auto should_process = [](const Node* node) {
        bool any_value = node->inputs_.empty();
        for (const auto* input : node->inputs_) {
            bool has = input->stream() && input->stream()->has_value();
            if (!has && !input->optional) return false;
            any_value = any_value || has;
        }
        return any_value;
    };
    auto execute = [&](size_t i) {
        Node* node = order_[i];
        if (should_process(node)) {
            node->timestamp_ = timestamp;
            node->process();
        }
        done[i] = true;
        finished.insert(node);
    };
    auto evaluate_pending = [&] {
        std::vector<Tensor> pending;
        for (const auto& node : nodes_) {
            for (auto* output : node->outputs_) output->stream()->collect_pending_tensors(pending);
        }
        for (const auto& input : inputs_) input->collect_pending_tensors(pending);
        if (!pending.empty()) {
            mlx::core::eval(pending);
            ++last_eval_count_;
        }
    };

    last_eval_count_ = 0;
    size_t remaining = n;
    for (auto& loopback : loopbacks_) {
        auto it = std::find(order_.begin(), order_.end(), loopback.node);
        done[it - order_.begin()] = true;
        --remaining;
    }

    while (remaining > 0) {
        bool progressed = false;
        // Build every lazy computation that can be built.
        for (size_t i = 0; i < n; ++i) {
            if (done[i] || order_[i]->reads_tensors() || !inputs_resolved(order_[i])) continue;
            execute(i);
            --remaining;
            progressed = true;
        }
        // Then evaluate everything pending at once and run the CPU readers
        // that are ready, in dependency order.
        bool evaluated = false;
        for (size_t i = 0; i < n; ++i) {
            if (done[i] || !order_[i]->reads_tensors() || !inputs_resolved(order_[i])) continue;
            if (!evaluated) {
                evaluate_pending();
                evaluated = true;
            }
            execute(i);
            --remaining;
            progressed = true;
            // A tensor reader may enable lazy builders; let them queue first so
            // their work joins the next batched evaluation.
            break;
        }
        if (!progressed) throw std::logic_error("Graph scheduling stalled");
    }

    for (auto& loopback : loopbacks_) loopback.store();
    for (const auto& input : inputs_) input->clear();
}

}  // namespace mlxpipe
