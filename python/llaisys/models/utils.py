from llaisys.libllaisys.llaisys_types import *
def torch_dtype_to_llaisys_dtype(config):
    torch_dtype = config["torch_dtype"]
    if torch_dtype == 'float32':
        return DataType.F32
    elif torch_dtype == 'float16':
        return DataType.F16
    elif torch_dtype == 'bfloat16':
        return DataType.BF16
    else:
        raise ValueError(f"Unsupported dtype: {torch_dtype}")