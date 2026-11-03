// Regression test for the TFLite runtime. Every model in the .task bundles
// runs on a fixed analytic input and is compared with golden outputs in
// tests/data/golden. The goldens were recorded after checking the runtime
// against the original hand-transpiled models (max |diff| 5e-5).
//
// Regenerate after an intended change with MLXPIPE_UPDATE_GOLDEN=1.
#include <cstdio>
#include <cstdlib>
#include <filesystem>

#include "mlxpipe/model/model_bundle.h"
#include "mlxpipe/model/tflite_model.h"
#include "test_util.h"

namespace mx = mlx::core;
using namespace mlxpipe;

static float MaxAbsDiff(const mx::array& a, const mx::array& b) {
    return mx::max(mx::abs(mx::subtract(mx::reshape(a, {-1}), mx::reshape(b, {-1})))).item<float>();
}

// Deterministic input in [0, scale] that does not depend on an RNG.
static mx::array AnalyticInput(const std::vector<int>& shape, float scale) {
    int n = 1;
    for (int d : shape) n *= d;
    mx::array i = mx::astype(mx::arange(n), mx::float32);
    mx::array v = mx::multiply(mx::add(mx::multiply(mx::sin(mx::multiply(i, mx::array(0.37f))), mx::array(0.5f)),
                                       mx::array(0.5f)),
                               mx::array(scale));
    return mx::reshape(v, mx::Shape(shape.begin(), shape.end()));
}

int main() {
    const std::string root = mlxpipe_test::source_dir();
    const bool update = std::getenv("MLXPIPE_UPDATE_GOLDEN") != nullptr;

    for (const std::string task : {"hand_landmarker.task", "face_landmarker.task"}) {
        ModelBundle bundle = ModelBundle::open(root + "/tools/models/" + task);
        for (const std::string& file : bundle.file_names()) {
            if (file.size() < 7 || file.substr(file.size() - 7) != ".tflite") continue;
            auto model = TfLiteModel::from_buffer(bundle.file(file), file);
            std::vector<int> shape = model->input_shape(0);
            // Blendshapes take landmark pixels; images take [0, 1].
            mx::array x = AnalyticInput(shape, shape.size() == 3 ? 256.0f : 1.0f);
            std::vector<mx::array> outputs = (*model)({x});
            mx::eval(outputs);

            // Compiled and eager execution agree.
            std::vector<mx::array> eager = model->run_uncompiled({x});
            for (size_t i = 0; i < outputs.size(); ++i) EXPECT_TRUE(MaxAbsDiff(outputs[i], eager[i]) < 1e-3f);

            // Image models: a batch of two equals two single runs.
            if (shape.size() == 4) {
                mx::array other = mx::subtract(mx::array(1.0f), x);
                mx::array batch = mx::concatenate({x, other}, 0);
                std::vector<mx::array> batched = (*model)({batch});
                std::vector<mx::array> second = (*model)({other});
                for (size_t i = 0; i < outputs.size(); ++i) {
                    mx::array expected = mx::concatenate({outputs[i], second[i]}, 0);
                    EXPECT_TRUE(MaxAbsDiff(batched[i], expected) < 1e-3f);
                }
            }

            const std::string golden = root + "/tests/data/golden/" + file.substr(0, file.size() - 7) + ".safetensors";
            if (update) {
                std::unordered_map<std::string, mx::array> named;
                for (size_t i = 0; i < outputs.size(); ++i) named.insert({std::to_string(i), outputs[i]});
                mx::save_safetensors(golden, named);
                std::printf("wrote %s\n", golden.c_str());
                continue;
            }
            EXPECT_TRUE(std::filesystem::exists(golden));
            if (!std::filesystem::exists(golden)) continue;
            auto expected = mx::load_safetensors(golden).first;
            for (size_t i = 0; i < outputs.size(); ++i) {
                float d = MaxAbsDiff(outputs[i], expected.at(std::to_string(i)));
                float scale = mx::max(mx::abs(expected.at(std::to_string(i)))).item<float>();
                std::printf("%-32s output %zu: max |diff| %.2e (scale %.2e)\n", file.c_str(), i, d, scale);
                EXPECT_TRUE(d <= 1e-4f * std::max(1.0f, scale));
            }
        }
    }
    TEST_MAIN_END();
}
