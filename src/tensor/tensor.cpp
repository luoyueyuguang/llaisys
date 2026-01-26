#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    // TO_BE_IMPLEMENTED();
    // Check if the tensor is contiguous
    // 比较 strides 是否等于每一层的元素大小的乘积
    ptrdiff_t expected_stride = 1;
    int ndim = static_cast<int>(this->ndim());
    for (int i = ndim - 1; i >= 0; i--) {
        if (this->strides()[i] != expected_stride) {
            return false;
        }
        expected_stride *= this->shape()[i];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // TO_BE_IMPLEMENTED();
    // 检查 order 是否合法
    CHECK_ARGUMENT(order.size() == this->ndim(), 
    "Tensor::permute: The order must have the same number of elements as the tensor's number of dimensions.");
    // 检查 order 是否合法
    std::vector<int8_t> visited(this->ndim(), 0);
    for (size_t i = 0; i < this->ndim(); i++) {
        CHECK_ARGUMENT(!visited[order[i]] && order[i] < this->ndim(), 
        "Tensor::permute: The order must be a permutation of the tensor's dimensions.");
        visited[order[i]] = 1;
    }
    // 计算新的 shape 和 strides
    std::vector<size_t> new_shape(this->ndim(), 0);
    std::vector<ptrdiff_t> new_strides(this->ndim(), 0);
    for (size_t i = 0; i < this->ndim(); i++) {
        new_shape[i] = this->shape()[order[i]];
        new_strides[i] = this->strides()[order[i]];
    }
    //创建新的 TensorMeta
    TensorMeta new_meta = this->_meta;
    new_meta.shape = new_shape;
    new_meta.strides = new_strides;
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // TO_BE_IMPLEMENTED();
    //检查是否是 contiguous
    CHECK_ARGUMENT(
        this -> isContiguous(),
        "Tensor::view: The tensor must be contiguous."
    );
    // 检查新的 shape 是否与原 shape 兼容
    size_t new_numel = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        new_numel *= shape[i];
    }
    CHECK_ARGUMENT(
        new_numel == this->numel(),
        "Tensor::view: The new shape must have the same number of elements as the original shape."
    );
    // 检查新的 shape 是否与原 shape 兼容
    // 计算新的 strides
    std::vector<ptrdiff_t> new_strides(shape.size(), 0);
    new_strides[shape.size() - 1] = 1;
    int shape_size = static_cast<int>(shape.size());
    for (int i = shape_size - 2; i >= 0; i--) {
        new_strides[i] = new_strides[i + 1] * shape[i + 1];
    }
    //创建新的 TensorMeta
    TensorMeta new_meta = this->_meta;
    new_meta.shape = std::move(shape);
    new_meta.strides = std::move(new_strides);

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // TO_BE_IMPLEMENTED();
    auto ndim = this->ndim();
    // 检查 dim 是否合法
    CHECK_ARGUMENT(
        dim < ndim,
        "Tensor::slice: The dimension must be less than the number of dimensions."
    );
    // 检查 start 和 end 是否合法
    CHECK_ARGUMENT(
        start <= end && end <= this->shape()[dim],
        "Tensor::slice: The start and end indices must be within the bounds of the dimension."
    );

    //new meta
    TensorMeta new_meta = this->_meta;
    std::vector<size_t> new_shape = this->shape();
    new_shape[dim] = end - start;
    new_meta.strides[dim] = this->strides()[dim];
    new_meta.shape = std::move(new_shape);

    //new offset
    ptrdiff_t new_offset = _offset + start * this->strides()[dim] * this->elementSize();
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    // TO_BE_IMPLEMENTED();
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        memcpy(this->data(), src_, this->numel() * this->elementSize());
    } else {
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src_,
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_H2D);
    }
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
