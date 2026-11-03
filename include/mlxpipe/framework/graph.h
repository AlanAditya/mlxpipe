// Code-first dataflow graphs, modelled on MediaPipe's CalculatorGraph.
//
// A Graph owns Nodes (MediaPipe "calculators"). Each node declares typed
// Input<T>/Output<T> ports as members; ports are connected in code:
//
//   mlxpipe::Graph graph;
//   auto image = graph.input<Image>("image");
//   auto& to_tensor = graph.add<ImageToTensorNode>(options);
//   image >> to_tensor.image;
//   auto& inference = graph.add<InferenceNode>(model);
//   to_tensor.tensors >> inference.tensors;
//   auto out = graph.output(inference.outputs);
//
//   image.send(frame);
//   graph.run(timestamp_us);
//   const auto& tensors = out.get();
//
// Execution is synchronous: run() processes one timestamp ("tick") through
// every node in dependency order. A stream carries at most one value per
// tick; a stream left empty plays the role of MediaPipe's empty packet, and
// a node whose required inputs are empty is skipped (its outputs stay empty),
// which is how gating (AllowIf / DisallowIf) propagates.
//
// MLX evaluates lazily, so nodes that only build tensor computations run
// first. Nodes that read tensor data on the CPU declare reads_tensors();
// before they run, every pending tensor in the graph is evaluated in a single
// mx::eval, batching independent model invocations into one GPU submission.
#pragma once

#include <mlx/mlx.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlxpipe {

using Tensor = mlx::core::array;
// Microseconds, like MediaPipe's Timestamp.
using Timestamp = int64_t;

class Graph;
class Node;

namespace internal {

struct StreamState {
    explicit StreamState(std::string stream_name) : name(std::move(stream_name)) {}
    virtual ~StreamState() = default;
    virtual bool has_value() const = 0;
    virtual void clear() = 0;
    // Appends tensors held by the current value that are not yet evaluated.
    virtual void collect_pending_tensors(std::vector<Tensor>& out) const = 0;

    std::string name;
    Node* producer = nullptr;  // null for graph inputs
};

template <typename T>
struct is_tensor_vector : std::false_type {};
template <>
struct is_tensor_vector<std::vector<Tensor>> : std::true_type {};

template <typename T>
struct TypedStreamState : StreamState {
    using StreamState::StreamState;
    bool has_value() const override { return value.has_value(); }
    void clear() override { value.reset(); }
    void collect_pending_tensors(std::vector<Tensor>& out) const override {
        if (!value) return;
        if constexpr (std::is_same_v<T, Tensor>) {
            if (!value->is_available()) out.push_back(*value);
        } else if constexpr (is_tensor_vector<T>::value) {
            for (const auto& t : *value) {
                if (!t.is_available()) out.push_back(t);
            }
        }
    }

    std::optional<T> value;
};

struct InputBase {
    virtual ~InputBase() = default;
    virtual const StreamState* stream() const = 0;
    std::string name;
    bool optional = false;
};

struct OutputBase {
    virtual ~OutputBase() = default;
    virtual StreamState* stream() = 0;
    std::string name;
};

}  // namespace internal

// A handle to a stream of values of type T. Cheap to copy; owned by the graph.
template <typename T>
class Stream {
public:
    Stream() = default;
    bool valid() const { return state_ != nullptr; }
    const std::string& name() const { return state_->name; }

private:
    explicit Stream(internal::TypedStreamState<T>* state) : state_(state) {}
    internal::TypedStreamState<T>* state_ = nullptr;

    friend class Graph;
    template <typename>
    friend class Input;
    template <typename>
    friend class Output;
    template <typename>
    friend class GraphInput;
    template <typename>
    friend class GraphOutput;
};

// The base class for graph nodes (MediaPipe calculators). Subclasses declare
// ports as members, initialised with `this`, and implement process().
class Node {
public:
    explicit Node(std::string type_name) : type_name_(std::move(type_name)) {}
    virtual ~Node() = default;
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    // Called once per tick when all required inputs have values.
    virtual void process() = 0;
    // True if process() reads tensor data on the CPU. The graph evaluates all
    // pending tensors before running such nodes.
    virtual bool reads_tensors() const { return false; }

    const std::string& type_name() const { return type_name_; }
    // Timestamp of the tick being processed.
    Timestamp timestamp() const { return timestamp_; }

private:
    std::string type_name_;
    Timestamp timestamp_ = 0;
    std::vector<internal::InputBase*> inputs_;
    std::vector<internal::OutputBase*> outputs_;

    friend class Graph;
    template <typename>
    friend class Input;
    template <typename>
    friend class Output;
};

// An input port. Required inputs must be connected and must carry a value
// for the node to run; optional inputs may be unconnected or empty.
template <typename T>
class Input : public internal::InputBase {
public:
    Input(Node* node, std::string port_name, bool is_optional = false) {
        name = std::move(port_name);
        optional = is_optional;
        node->inputs_.push_back(this);
    }

    void connect(Stream<T> stream) {
        if (!stream.valid()) throw std::invalid_argument("Connecting invalid stream to input '" + name + "'");
        state_ = stream.state_;
    }
    bool connected() const { return state_ != nullptr; }
    bool has() const { return state_ && state_->value.has_value(); }
    const T& get() const {
        if (!has()) throw std::logic_error("Input '" + name + "' has no value");
        return *state_->value;
    }
    const T& operator*() const { return get(); }
    const T* operator->() const { return &get(); }

    const internal::StreamState* stream() const override { return state_; }

private:
    internal::TypedStreamState<T>* state_ = nullptr;
};

// An output port. Each output owns the stream it produces.
template <typename T>
class Output : public internal::OutputBase {
public:
    Output(Node* node, std::string port_name)
        : state_(std::make_unique<internal::TypedStreamState<T>>(node->type_name() + "." + port_name)) {
        name = std::move(port_name);
        state_->producer = node;
        node->outputs_.push_back(this);
    }

    void send(T value) { state_->value = std::move(value); }
    Stream<T> stream() const { return Stream<T>(state_.get()); }
    operator Stream<T>() const { return stream(); }

    internal::StreamState* stream() override { return state_.get(); }

private:
    std::unique_ptr<internal::TypedStreamState<T>> state_;
};

template <typename T>
void operator>>(Stream<T> stream, Input<T>& input) {
    input.connect(stream);
}
template <typename T>
void operator>>(const Output<T>& output, Input<T>& input) {
    input.connect(output.stream());
}

template <typename T>
class GraphInput;
template <typename T>
void operator>>(const GraphInput<T>& graph_input, Input<T>& input) {
    input.connect(graph_input.stream());
}

// A value fed into the graph from outside. send() sets the value for the next
// run(); it is cleared afterwards.
template <typename T>
class GraphInput {
public:
    GraphInput() = default;
    void send(T value) { state_->value = std::move(value); }
    Stream<T> stream() const { return Stream<T>(state_); }
    operator Stream<T>() const { return stream(); }

private:
    explicit GraphInput(internal::TypedStreamState<T>* state) : state_(state) {}
    internal::TypedStreamState<T>* state_ = nullptr;
    friend class Graph;
};

// Reads a stream's value after run(). Values stay valid until the next run().
template <typename T>
class GraphOutput {
public:
    GraphOutput() = default;
    bool has() const { return state_->value.has_value(); }
    const T& get() const {
        if (!has()) throw std::logic_error("Graph output '" + state_->name + "' has no value");
        return *state_->value;
    }
    // The value, or `fallback` when the stream was empty this tick.
    T value_or(T fallback) const { return has() ? *state_->value : std::move(fallback); }

private:
    explicit GraphOutput(internal::TypedStreamState<T>* state) : state_(state) {}
    internal::TypedStreamState<T>* state_ = nullptr;
    friend class Graph;
};

// MediaPipe's PreviousLoopbackCalculator: `previous` carries the value that
// was sent to `next` on the previous tick. On the first tick, and after a
// tick where `next` was empty, it carries `initial` (if given).
template <typename T>
class LoopbackNode : public Node {
public:
    explicit LoopbackNode(std::optional<T> initial)
        : Node("PreviousLoopback"), next(this, "next", /*is_optional=*/true), previous(this, "previous"),
          initial_(initial), stored_(std::move(initial)) {}

    Input<T> next;
    Output<T> previous;

    void process() override {}

private:
    void emit() {
        if (stored_) previous.send(*stored_);
    }
    void store() { stored_ = next.has() ? std::optional<T>(next.get()) : initial_; }

    std::optional<T> initial_;
    std::optional<T> stored_;
    friend class Graph;
};

class Graph {
public:
    Graph() = default;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    // Adds a node constructed from `args` and returns a reference to it.
    template <typename N, typename... Args>
    N& add(Args&&... args) {
        static_assert(std::is_base_of_v<Node, N>, "Graph::add requires a Node subclass");
        auto node = std::make_unique<N>(std::forward<Args>(args)...);
        N& ref = *node;
        nodes_.push_back(std::move(node));
        prepared_ = false;
        return ref;
    }

    template <typename T>
    GraphInput<T> input(std::string name) {
        auto state = std::make_unique<internal::TypedStreamState<T>>(std::move(name));
        GraphInput<T> handle(state.get());
        inputs_.push_back(std::move(state));
        return handle;
    }

    template <typename T>
    GraphOutput<T> output(Stream<T> stream) {
        return GraphOutput<T>(stream.state_);
    }
    template <typename T>
    GraphOutput<T> output(const Output<T>& port) {
        return output(port.stream());
    }

    // A back edge: connect a stream to `.next`, read last tick's value from
    // `.previous`.
    template <typename T>
    LoopbackNode<T>& loopback(std::optional<T> initial = std::nullopt) {
        auto& node = add<LoopbackNode<T>>(std::move(initial));
        loopbacks_.push_back({&node, [&node] { node.emit(); }, [&node] { node.store(); }});
        return node;
    }

    // Processes one tick. Throws if the graph is malformed or a node throws.
    void run(Timestamp timestamp);

    // Number of GPU synchronisations in the last run() (for profiling).
    int last_eval_count() const { return last_eval_count_; }

private:
    struct LoopbackHooks {
        Node* node;
        std::function<void()> emit;
        std::function<void()> store;
    };

    void prepare();
    bool is_loopback(const Node* node) const;

    std::vector<std::unique_ptr<Node>> nodes_;
    std::vector<std::unique_ptr<internal::StreamState>> inputs_;
    std::vector<LoopbackHooks> loopbacks_;
    std::vector<Node*> order_;  // topological order
    bool prepared_ = false;
    int last_eval_count_ = 0;
};

}  // namespace mlxpipe
