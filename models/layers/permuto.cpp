#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include "permutohedral.h"

/**
 *
 * @param input_values  Input values to filter (e.g. Q distributions). Has shape (channels, height, width)
 * @param features      Features for the permutohedral lattice. Has shape (height, width, feature_channels). Note that
 *                      channels are at the end!
 * @return Filtered values with shape (channels, height, width)
 */
std::vector<at::Tensor> permuto_forward(torch::Tensor input_values, torch::Tensor features) {

    auto input_sizes = input_values.sizes();  // (channels, d, w, height)
    auto feature_sizes = features.sizes();  // (d, w, height, num_features)

    auto d = feature_sizes[0];
    auto w = feature_sizes[1];
    auto h = feature_sizes[2];
    auto n_feature_dims = static_cast<int>(feature_sizes[3]);
    auto n_pixels = static_cast<int>(d * w * h);
    auto n_channels = static_cast<int>(input_sizes[0]); // 1 for 3D

    // Validate the arguments
    if (input_sizes[1] != d || input_sizes[2] != w || input_sizes[3] != h) {
        throw std::runtime_error("Sizes of `input_values` and `features` do not match!");
    }

    if (!(input_values.dtype() == torch::kFloat32)) {
        throw std::runtime_error("`input_values` must have float32 type.");
    }

    if (!(features.dtype() == torch::kFloat32)) {
        throw std::runtime_error("`features` must have float32 type.");
    }

    // Create the output tensor
    auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .layout(torch::kStrided)
            .device(torch::kCPU)
            .requires_grad(false);

    auto output_values = torch::empty(input_sizes, options);
    output_values = output_values.contiguous();

    Permutohedral p;
//    std::cout << "p newed" << std::endl;
    p.init(features.contiguous().data<float>(), n_feature_dims, n_pixels);
//    std::cout << "p initialized" << std::endl;
    p.compute(output_values.data<float>(), input_values.contiguous().data<float>(), n_channels);
//    std::cout << "p computed" << std::endl;

//    std::cout << "output_values" << output_values << std::endl;
    return {output_values};
}


std::vector<at::Tensor> permuto_backward(torch::Tensor grads, torch::Tensor features) {

    auto grad_sizes = grads.sizes();  // (channels, d, w, height)
    auto feature_sizes = features.sizes();  // (d, w, height, num_features)

    auto d = feature_sizes[0];
    auto w = feature_sizes[1];
    auto h = feature_sizes[2];
    auto n_feature_dims = static_cast<int>(feature_sizes[3]);
    auto n_pixels = static_cast<int>(d * h * w);
    auto n_channels = static_cast<int>(grad_sizes[0]);

    // Validate the arguments
    if (grad_sizes[1] != d || grad_sizes[2] != w || grad_sizes[3] != h) {
        throw std::runtime_error("Sizes of `grad_values` and `features` do not match!");
    }

    if (!(grads.dtype() == torch::kFloat32)) {
        throw std::runtime_error("`input_values` must have float32 type.");
    }

    if (!(features.dtype() == torch::kFloat32)) {
        throw std::runtime_error("`features` must have float32 type.");
    }

    // Create the output tensor
    auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .layout(torch::kStrided)
            .device(torch::kCPU)
            .requires_grad(false);

    auto grads_back = torch::empty(grad_sizes, options);
    grads_back = grads_back.contiguous();

    Permutohedral p;
//    std::cout << "p back newed" << std::endl;
    p.init(features.contiguous().data<float>(), n_feature_dims, n_pixels);
//    std::cout << "p back initialized" << std::endl;
    p.compute(grads_back.data<float>(), grads.contiguous().data<float>(), n_channels, true);
//    std::cout << "p back computed" << std::endl;

    return {grads_back};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &permuto_forward, "PERMUTO forward");
    m.def("backward", &permuto_backward, "PERMUTO backward");
}
