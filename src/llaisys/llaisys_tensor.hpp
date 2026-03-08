#pragma once
#include "llaisys/tensor.h"

#include "../tensor/tensor.hpp"

LLAISYS_EXTERN_BEGIN
    typedef struct LlaisysTensor {
        llaisys::tensor_t tensor;
    } LlaisysTensor;
LLAISYS_EXTERN_END
