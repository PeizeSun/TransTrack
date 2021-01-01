#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>


at::Tensor
ms_deform_attn_cpu_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
{
    AT_ERROR("Not implement on cpu");
}

std::vector<at::Tensor>
ms_deform_attn_cpu_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step)
{
    AT_ERROR("Not implement on cpu");
}

