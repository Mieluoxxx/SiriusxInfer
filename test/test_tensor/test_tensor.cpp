#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tensor/tensor.h"

TEST(test_tensor, create_cube) {
    using namespace infer;
    int32_t size = 27;
    std::vector<float> datas;
    for (int i = 0; i < size; i++) {
        datas.push_back(float(i));
    }
    arma::Cube<float> cube(3, 3, 3);
    memcpy(cube.memptr(), datas.data(), size * sizeof(float));
    LOG(INFO) << cube;
}