#include "tensor/tensor.h"

#include <glog/logging.h>

namespace infer {
Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
    _data = arma::fcube(rows, cols, channels);
    if (channels == 1 && rows == 1) {
        this->_raw_shapes = std::vector<uint32_t>{cols};
    } else if (channels == 1) {
        this->_raw_shapes = std::vector<uint32_t>{rows, cols};
    } else {
        this->_raw_shapes = std::vector<uint32_t>{channels, rows, cols};
    }
}

Tensor<float>::Tensor(uint32_t size) {
    // armadillo 是列主序
    _data = arma::fcube(1, size, 1);
    this->_raw_shapes = std::vector<uint32_t>{size};
}

Tensor<float>::Tensor(uint32_t rows, uint32_t cols) {
    _data = arma::fcube(rows, cols, 1);
    this->_raw_shapes = std::vector<uint32_t>{rows, cols};
}

Tensor<float>::Tensor(const std::vector<uint32_t>& shapes) {
    CHECK(!shapes.empty() && shapes.size() <= 3);

    uint32_t remaining = 3 - shapes.size();
    std::vector<uint32_t> shapes_(3, 1);
    std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

    uint32_t channels = shapes_.at(0);
    uint32_t rows = shapes_.at(1);
    uint32_t cols = shapes_.at(2);

    _data = arma::fcube(rows, cols, channels);
    if (channels == 1 && rows == 1) {
        this->_raw_shapes = std::vector<uint32_t>{cols};
    } else if (channels == 1) {
        this->_raw_shapes = std::vector<uint32_t>{rows, cols};
    } else {
        this->_raw_shapes = std::vector<uint32_t>{channels, rows, cols};
    }
}

Tensor<float>::Tensor(const Tensor& tensor) {
    if (this != &tensor) {
        this->_data = tensor._data;
        this->_raw_shapes = tensor._raw_shapes;
    }
}

Tensor<float>::Tensor(Tensor<float>&& tensor) noexcept {
    if (this != &tensor) {
        this->_data = std::move(tensor._data);
        this->_raw_shapes = tensor._raw_shapes;
    }
}

Tensor<float>& Tensor<float>::operator=(Tensor<float>&& tensor) noexcept {
    if (this != &tensor) {
        this->_data = std::move(tensor._data);
        this->_raw_shapes = tensor._raw_shapes;
    }
    return *this;
}

Tensor<float>& Tensor<float>::operator=(const Tensor& tensor) {
    if (this != &tensor) {
        this->_data = tensor._data;
        this->_raw_shapes = tensor._raw_shapes;
    }
    return *this;
}

uint32_t Tensor<float>::rows() const {
    CHECK(!this->_data.empty());
    return this->_data.n_rows;
}

uint32_t Tensor<float>::cols() const {
    CHECK(!this->_data.empty());
    return this->_data.n_cols;
}

uint32_t Tensor<float>::channels() const {
    CHECK(!this->_data.empty());
    return this->_data.n_slices;
}

uint32_t Tensor<float>::size() const {
    CHECK(!this->_data.empty());
    return this->_data.size();
}

void Tensor<float>::set_data(const arma::fcube& data) {
    CHECK(data.n_rows == this->_data.n_rows)
        << data.n_rows << " != " << this->_data.n_rows;
    CHECK(data.n_cols == this->_data.n_cols)
        << data.n_cols << " != " << this->_data.n_cols;
    CHECK(data.n_slices == this->_data.n_slices)
        << data.n_slices << " != " << this->_data.n_slices;
    this->_data = data;
}

bool Tensor<float>::empty() const { return this->_data.empty(); }

float Tensor<float>::index(uint32_t offset) const {
    CHECK(offset < this->_data.size()) << "Tensor index out of bound!";
    return this->_data.at(offset);
}

float& Tensor<float>::index(uint32_t offset) {
    CHECK(offset < this->_data.size()) << "Tensor index out of bound!";
    return this->_data.at(offset);
}

std::vector<uint32_t> Tensor<float>::shapes() const {
    CHECK(!this->_data.empty());
    return {this->channels(), this->rows(), this->cols()};
}

arma::fcube& Tensor<float>::data() { return this->_data; }

const arma::fcube& Tensor<float>::data() const { return this->_data; }

arma::fmat& Tensor<float>::slice(uint32_t channel) {
    CHECK_LT(channel, this->channels());
    return this->_data.slice(channel);
}

const arma::fmat& Tensor<float>::slice(uint32_t channel) const {
    CHECK_LT(channel, this->channels());
    return this->_data.slice(channel);
}

float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
    CHECK_LT(row, this->rows());
    CHECK_LT(col, this->cols());
    CHECK_LT(channel, this->channels());
    return this->_data.at(row, col, channel);
}

float& Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
    CHECK_LT(row, this->rows());
    CHECK_LT(col, this->cols());
    CHECK_LT(channel, this->channels());
    return this->_data.at(row, col, channel);
}

void Tensor<float>::Padding(const std::vector<uint32_t>& pads,
                            float padding_value) {
    CHECK(!this->_data.empty());
    CHECK_EQ(pads.size(), 4);
    // 四周填充的维度
    uint32_t pad_rows1 = pads.at(0);  // up
    uint32_t pad_rows2 = pads.at(1);  // bottom
    uint32_t pad_cols1 = pads.at(2);  // left
    uint32_t pad_cols2 = pads.at(3);  // right

    arma::Cube<float> new_data(this->_data.n_rows + pad_rows1 + pad_rows2,
                               this->_data.n_cols + pad_cols1 + pad_cols2,
                               this->_data.n_slices);

    new_data.fill(padding_value);

    // subcube
    // 函数用于获取一个子立方体，它接受六个参数，分别是起始行、起始列、起始切片、结束行、结束列、结束切片
    new_data.subcube(pad_rows1, pad_cols1, 0, new_data.n_rows - pad_rows2 - 1,
                     new_data.n_cols - pad_cols2 - 1, new_data.n_slices - 1) =
        this->_data;

    this->_data = std::move(new_data);
    this->_raw_shapes =
        std::vector<uint32_t>{this->channels(), this->rows(), this->cols()};
}

void Tensor<float>::Fill(float value) {
    CHECK(!this->_data.empty());
    this->_data.fill(value);
}

void Tensor<float>::Fill(const std::vector<float>& values, bool row_major) {
    CHECK(!this->_data.empty());
    const uint32_t total_elems = this->_data.size();
    CHECK_EQ(values.size(), total_elems);
    if (row_major) {
        const uint32_t rows = this->rows();
        const uint32_t cols = this->cols();
        const uint32_t planes = rows * cols;
        const uint32_t channels = this->_data.n_slices;

        for (uint32_t i = 0; i < channels; ++i) {
            auto& channel_data = this->_data.slice(i);
            const arma::fmat& channel__datat = arma::fmat(
                values.data() + i * planes, this->cols(), this->rows());
            channel_data = channel__datat.t();
        }
    } else {
        std::copy(values.begin(), values.end(), this->_data.memptr());
    }
}
void Tensor<float>::Show() {
    for (uint32_t i = 0; i < this->channels(); ++i) {
        LOG(INFO) << "Channel: " << i;
        LOG(INFO) << "\n" << this->_data.slice(i);
    }
}

void Tensor<float>::Flatten(bool row_major) {
    CHECK(!this->_data.empty());
    const uint32_t size = this->_data.size();
    this->Reshape({size}, row_major);
}

void Tensor<float>::Rand() {
    CHECK(!this->_data.empty());
    this->_data.randn();
}

void Tensor<float>::Ones() {
    CHECK(!this->_data.empty());
    this->Fill(1.f);
}

// 接受一个 float 参数并返回一个 float 值
void Tensor<float>::Transform(const std::function<float(float)>& filter) {
    CHECK(!this->_data.empty());  // 检查张量数据是否为空
    this->_data.transform(filter);  // 对张量中的每个元素应用变换函数
}

const std::vector<uint32_t>& Tensor<float>::raw_shapes() const {
    CHECK(!this->_raw_shapes.empty());
    CHECK_LE(this->_raw_shapes.size(), 3);
    CHECK_GE(this->_raw_shapes.size(), 1);
    return this->_raw_shapes;
}

void Tensor<float>::Reshape(const std::vector<uint32_t>& shapes,
                            bool row_major) {
    CHECK(!this->_data.empty());
    CHECK(!shapes.empty());
    const uint32_t origin_size = this->size();
    const uint32_t current_size =
        std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies());
    CHECK(shapes.size() <= 3);
    CHECK(current_size == origin_size);

    std::vector<float> values;
    if (row_major) {
        values = this->values(true);
    }
    if (shapes.size() == 3) {
        this->_data.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
        this->_raw_shapes = {shapes.at(0), shapes.at(1), shapes.at(2)};
    } else if (shapes.size() == 2) {
        this->_data.reshape(shapes.at(0), shapes.at(1), 1);
        this->_raw_shapes = {shapes.at(0), shapes.at(1)};
    } else {
        this->_data.reshape(1, shapes.at(0), 1);
        this->_raw_shapes = {shapes.at(0)};
    }

    if (row_major) {
        this->Fill(values, true);
    }
}

float* Tensor<float>::raw_ptr() {
    CHECK(!this->_data.empty());
    return this->_data.memptr();
}

float* Tensor<float>::raw_ptr(uint32_t offset) {
    const uint32_t size = this->size();
    CHECK(!this->_data.empty());
    CHECK_LT(offset, size);
    return this->_data.memptr() + offset;
}

std::vector<float> Tensor<float>::values(bool row_major) {
    /**
     * 列主序         行主序
     * 0 3 6         0 1 2
     * 1 4 7         3 4 5
     * 2 5 8         6 7 8
     */
    CHECK_EQ(this->_data.empty(), false);
    std::vector<float> values(this->_data.size());

    if (!row_major) {  // 列主序
        std::copy(this->_data.mem, this->_data.mem + this->_data.size(),
                  values.begin());
    } else {  // 行主序
        uint32_t index = 0;
        for (uint32_t c = 0; c < this->_data.n_slices; ++c) {
            // 获取当前通道并转置
            const arma::fmat& channel = this->_data.slice(c).t();
            // 复制转置后的数据到结果向量
            std::copy(channel.begin(), channel.end(), values.begin() + index);
            index += channel.size();
        }
        CHECK_EQ(index, values.size());
    }
    return values;
}

float* Tensor<float>::matrix_raw_ptr(uint32_t index) {
    CHECK_LT(index, this->channels());
    uint32_t offset = index * this->rows() * this->cols();  // 计算内存偏移量
    CHECK_LE(offset, this->size());
    float* mem_ptr = this->raw_ptr() + offset;  // 获取基地址后偏移
    return mem_ptr;  // 返回指定通道的裸指针
}
}  // namespace infer