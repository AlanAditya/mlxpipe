#include <mlx/mlx.h>
#include <iostream>

int main() {
    auto weights = mlx::core::load("tools/models/palm_detection_full.safetensors");
    std::cout << "Loaded " << weights.size() << " weights from safetensors!" << std::endl;
    return 0;
}
