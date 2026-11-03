// Runs TensorFlow Lite models on MLX.
//
// The .tflite FlatBuffer is read directly (no TensorFlow dependency). At load
// time constant subgraphs (such as float16 weight dequantization) are folded,
// and weights are laid out once in MLX's convolution format. The forward pass
// is compiled with mx::compile, which fuses the elementwise ops between
// convolutions; calls with a new input shape trigger one retrace.
//
// Inputs and outputs use the model's layout (NHWC for images). The leading
// dimension is treated as the batch, so a model exported with batch 1 can
// run on [N, H, W, C] inputs.
#pragma once

#include <mlx/mlx.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <string>
#include <vector>

namespace mlxpipe {

class TfLiteModel {
public:
    // Loads a .tflite file. Throws std::runtime_error for unreadable files and
    // for operators this runtime does not implement.
    static std::shared_ptr<TfLiteModel> load(const std::string& path);
    static std::shared_ptr<TfLiteModel> from_buffer(std::vector<uint8_t> data, std::string name = "model");

    ~TfLiteModel();
    TfLiteModel(const TfLiteModel&) = delete;
    TfLiteModel& operator=(const TfLiteModel&) = delete;

    // Builds the (lazy) outputs for `inputs`. Nothing is computed until the
    // outputs are evaluated.
    std::vector<mlx::core::array> operator()(const std::vector<mlx::core::array>& inputs) const;
    // The same computation without mx::compile; useful for debugging.
    std::vector<mlx::core::array> run_uncompiled(const std::vector<mlx::core::array>& inputs) const;

    const std::string& name() const;
    size_t num_inputs() const;
    size_t num_outputs() const;
    // Shapes as declared in the model (batch dimension included).
    std::vector<int> input_shape(size_t index) const;
    std::vector<int> output_shape(size_t index) const;
    const std::string& output_name(size_t index) const;

    // Input normalization from the model's TFLite metadata
    // (NormalizationOptions): tensor = (pixel - mean) / std.
    struct Normalization {
        float mean = 0.0f;
        float std = 1.0f;
    };
    // Empty when the model has no normalization metadata.
    const std::optional<Normalization>& input_normalization() const;
    // The tensor value range for pixels in [0, 255], as MediaPipe's
    // ImagePreprocessingGraph derives it; `fallback` without metadata.
    std::pair<float, float> input_range(std::pair<float, float> fallback = {0.0f, 1.0f}) const;

private:
    struct Impl;
    explicit TfLiteModel(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    std::function<std::vector<mlx::core::array>(const std::vector<mlx::core::array>&)> compiled_;
};

}  // namespace mlxpipe
