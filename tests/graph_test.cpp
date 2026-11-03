#include "mlxpipe/framework/graph.h"
#include "test_util.h"

namespace mx = mlx::core;
using namespace mlxpipe;

struct AddNode : Node {
    AddNode() : Node("Add") {}
    Input<int> a{this, "a"};
    Input<int> b{this, "b", /*is_optional=*/true};
    Output<int> sum{this, "sum"};
    void process() override { sum.send(a.get() + (b.has() ? b.get() : 0)); }
};

// Lazily multiplies a tensor; never reads tensor data.
struct ScaleNode : Node {
    explicit ScaleNode(float s) : Node("Scale"), scale(s) {}
    Input<Tensor> in{this, "in"};
    Output<Tensor> out{this, "out"};
    float scale;
    void process() override { out.send(mx::multiply(in.get(), mx::array(scale))); }
};

// Reads tensor data on the CPU.
struct ReadNode : Node {
    ReadNode() : Node("Read") {}
    Input<Tensor> in{this, "in"};
    Output<float> value{this, "value"};
    bool reads_tensors() const override { return true; }
    void process() override {
        EXPECT_TRUE(in.get().is_available());
        value.send(in.get().item<float>());
    }
};

int main() {
    // Basic wiring, optional inputs and skipping on empty required inputs.
    {
        Graph g;
        auto a = g.input<int>("a");
        auto b = g.input<int>("b");
        auto& add = g.add<AddNode>();
        a >> add.a;
        b >> add.b;
        auto out = g.output(add.sum);
        a.send(2);
        b.send(3);
        g.run(0);
        EXPECT_EQ(out.get(), 5);
        a.send(4);  // b empty: optional
        g.run(1);
        EXPECT_EQ(out.get(), 4);
        b.send(1);  // a empty: required, node skipped
        g.run(2);
        EXPECT_TRUE(!out.has());
    }
    // Loopback: running sum across ticks.
    {
        Graph g;
        auto x = g.input<int>("x");
        auto& prev = g.loopback<int>(0);
        auto& add = g.add<AddNode>();
        x >> add.a;
        prev.previous >> add.b;
        add.sum >> prev.next;
        auto out = g.output(add.sum);
        for (int i = 1; i <= 4; ++i) {
            x.send(i);
            g.run(i);
        }
        EXPECT_EQ(out.get(), 10);
    }
    // Two independent branches of lazy work are evaluated in one sync.
    {
        Graph g;
        auto t = g.input<Tensor>("t");
        auto& s1 = g.add<ScaleNode>(2.0f);
        auto& s2 = g.add<ScaleNode>(3.0f);
        auto& r1 = g.add<ReadNode>();
        auto& r2 = g.add<ReadNode>();
        t >> s1.in;
        t >> s2.in;
        s1.out >> r1.in;
        s2.out >> r2.in;
        auto o1 = g.output(r1.value);
        auto o2 = g.output(r2.value);
        t.send(mx::array(1.5f));
        g.run(0);
        EXPECT_NEAR(o1.get(), 3.0, 1e-6);
        EXPECT_NEAR(o2.get(), 4.5, 1e-6);
        EXPECT_EQ(g.last_eval_count(), 1);
    }
    // Unconnected required input is reported.
    {
        Graph g;
        g.add<AddNode>();
        bool threw = false;
        try {
            g.run(0);
        } catch (const std::logic_error&) {
            threw = true;
        }
        EXPECT_TRUE(threw);
    }
    TEST_MAIN_END();
}
