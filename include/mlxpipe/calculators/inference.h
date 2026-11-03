// MLX replacement for mediapipe/calculators/tensor/inference_calculator.cc.
#pragma once

#include <memory>
#include <vector>

#include "mlxpipe/framework/graph.h"
#include "mlxpipe/model/tflite_model.h"

namespace mlxpipe {

// Runs a TFLite model on MLX. The outputs are lazy: the graph evaluates them
// together with any other pending work before a node reads them.
class InferenceNode : public Node {
public:
    explicit InferenceNode(std::shared_ptr<const TfLiteModel> model)
        : Node("Inference"), model_(std::move(model)) {}

    Input<Tensor> tensor{this, "tensor"};
    Output<std::vector<Tensor>> tensors{this, "tensors"};

    void process() override { tensors.send((*model_)({tensor.get()})); }

private:
    std::shared_ptr<const TfLiteModel> model_;
};

// Runs a model once per input tensor (for models that only support batch 1)
// and stacks each output along the batch dimension. All invocations are lazy,
// so they still share one GPU submission.
class PerItemInferenceNode : public Node {
public:
    explicit PerItemInferenceNode(std::shared_ptr<const TfLiteModel> model)
        : Node("PerItemInference"), model_(std::move(model)) {}

    Input<std::vector<Tensor>> items{this, "items"};
    Output<std::vector<Tensor>> tensors{this, "tensors"};

    void process() override {
        if (items.get().empty()) return;
        std::vector<std::vector<Tensor>> per_output;
        for (const auto& item : items.get()) {
            std::vector<Tensor> outputs = (*model_)({item});
            per_output.resize(outputs.size());
            for (size_t i = 0; i < outputs.size(); ++i) {
                Tensor t = outputs[i];
                if (t.ndim() == 0 || t.shape(0) != 1) t = mlx::core::expand_dims(t, 0);
                per_output[i].push_back(t);
            }
        }
        std::vector<Tensor> stacked;
        for (auto& parts : per_output) stacked.push_back(mlx::core::concatenate(parts, 0));
        tensors.send(std::move(stacked));
    }

private:
    std::shared_ptr<const TfLiteModel> model_;
};

}  // namespace mlxpipe
