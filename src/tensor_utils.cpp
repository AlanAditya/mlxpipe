#include "tensor_utils.h"
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace mlx_vision {

mlx::core::array cvMat_to_mlxArray(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty.");
    }
    
    cv::Mat rgb_image;
    // Convert BGR to RGB
    if (image.channels() == 3) {
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    } else {
        rgb_image = image.clone(); // Fallback if already 1 channel or unknown
    }
    
    // Convert uint8 to float32
    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);
    
    // We create the MLX array directly from the float32 data pointer.
    // The shape is [Height, Width, Channels]
    std::vector<int> shape = {float_image.rows, float_image.cols, float_image.channels()};
    
    // Create the mlx array
    // Note: mlx::core::array copies the data by default when initialized from a pointer.
    mlx::core::array tensor(
        reinterpret_cast<float*>(float_image.data), 
        shape, 
        mlx::core::float32
    );
    
    // Force copy of the data by performing a dummy operation
    // This is necessary because the pointer-based array constructor does not copy data
    // and float_image will be destroyed at the end of this function!
    tensor = mlx::core::add(tensor, mlx::core::array(0.0f));
    mlx::core::eval(tensor);
    
    return tensor;
}

cv::Mat mlxArray_to_cvMat(const mlx::core::array& tensor) {
    // Ensure the tensor is evaluated and available in host memory
    mlx::core::eval(tensor);
    
    if (tensor.ndim() != 3) {
        throw std::invalid_argument("Expected a 3D tensor for image conversion.");
    }
    
    int rows = tensor.shape(0);
    int cols = tensor.shape(1);
    int channels = tensor.shape(2);
    
    if (tensor.dtype() != mlx::core::float32) {
        throw std::invalid_argument("Expected float32 tensor.");
    }
    
    // Create a continuous cv::Mat and copy the data from the tensor
    cv::Mat float_image(rows, cols, CV_32FC(channels));
    
    // Getting the raw pointer from an MLX array
    const float* data_ptr = tensor.data<float>();
    std::memcpy(float_image.data, data_ptr, tensor.nbytes());
    
    // Convert back to uint8 [0, 255]
    cv::Mat uint8_image;
    float_image.convertTo(uint8_image, CV_8UC(channels), 255.0);
    
    // Convert RGB back to BGR
    cv::Mat bgr_image;
    if (channels == 3) {
        cv::cvtColor(uint8_image, bgr_image, cv::COLOR_RGB2BGR);
    } else {
        bgr_image = uint8_image;
    }
    
    return bgr_image;
}

} // namespace mlx_vision
