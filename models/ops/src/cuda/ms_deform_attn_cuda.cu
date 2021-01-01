#include <vector>
#include "cuda/ms_deform_im2col_cuda.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// #include <THC/THC.h>
// #include <THC/THCAtomics.cuh>
// #include <THC/THCDeviceUtils.cuh>

// extern THCState *state;

// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu


at::Tensor ms_deform_attn_cuda_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
    // value: N_, S_, M_, D_
    // spatial_shapes: L_, 2
    // sampling_loc: N_, Lq_, M_, L_, P_, 2
{
    AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");

    AT_ASSERTM(value.type().is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(spatial_shapes.type().is_cuda(), "spatial_shapes must be a CUDA tensor");
    AT_ASSERTM(sampling_loc.type().is_cuda(), "sampling_loc must be a CUDA tensor");
    AT_ASSERTM(attn_weight.type().is_cuda(), "attn_weight must be a CUDA tensor");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);
    
    auto output = at::empty({batch, num_query, num_heads, channels}, value.options());

    auto level_start_index = at::zeros({num_levels}, spatial_shapes.options());
    for (int lvl = 1; lvl < num_levels; ++lvl)
    {
        auto shape_prev = spatial_shapes.select(0, lvl-1);
        auto size_prev =  at::mul(shape_prev.select(0, 0), shape_prev.select(0, 1));
        level_start_index.select(0, lvl) = at::add(level_start_index.select(0, lvl-1), size_prev);
    }

    // define alias for easy use
    const int batch_n = im2col_step_;
    auto output_n = output.view({batch/im2col_step_, batch_n, num_query, num_heads, channels});
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        auto columns = at::empty({num_levels*num_point, batch_n, num_query, num_heads, channels}, value.options());
        AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_forward_cuda", ([&] {
            ms_deformable_im2col_cuda(at::cuda::getCurrentCUDAStream(),
                value.data<scalar_t>() + n * im2col_step_ * per_value_size,
                spatial_shapes.data<int64_t>(),
                level_start_index.data<int64_t>(),
                sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                columns.data<scalar_t>());

        }));
        output_n.select(0, n) = at::sum(columns, 0);
    }

    output = output.view({batch, num_query, num_heads*channels});

    return output;
}


std::vector<at::Tensor> ms_deform_attn_cuda_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step)
{

    AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");

    AT_ASSERTM(value.type().is_cuda(), "value must be a CUDA tensor");
    AT_ASSERTM(spatial_shapes.type().is_cuda(), "spatial_shapes must be a CUDA tensor");
    AT_ASSERTM(sampling_loc.type().is_cuda(), "sampling_loc must be a CUDA tensor");
    AT_ASSERTM(attn_weight.type().is_cuda(), "attn_weight must be a CUDA tensor");

    const int batch = value.size(0);
    const int spatial_size = value.size(1);
    const int num_heads = value.size(2);
    const int channels = value.size(3);

    const int num_levels = spatial_shapes.size(0);

    const int num_query = sampling_loc.size(1);
    const int num_point = sampling_loc.size(4);

    const int im2col_step_ = std::min(batch, im2col_step);

    AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)", batch, im2col_step_);

    auto grad_value = at::zeros_like(value);
    auto grad_sampling_loc = at::zeros_like(sampling_loc);
    auto grad_attn_weight = at::zeros_like(attn_weight);

    auto level_start_index = at::zeros({num_levels}, spatial_shapes.options());
    for (int lvl = 1; lvl < num_levels; ++lvl)
    {
        auto shape_prev = spatial_shapes.select(0, lvl-1);
        auto size_prev =  at::mul(shape_prev.select(0, 0), shape_prev.select(0, 1));
        level_start_index.select(0, lvl) = at::add(level_start_index.select(0, lvl-1), size_prev);
    }

    const int batch_n = im2col_step_;
    auto per_value_size = spatial_size * num_heads * channels;
    auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
    auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
    auto grad_output_n = grad_output.view({batch/im2col_step_, batch_n, num_query, num_heads, channels});
    for (int n = 0; n < batch/im2col_step_; ++n)
    {
        auto grad_output_g = grad_output_n.select(0, n);
        AT_DISPATCH_FLOATING_TYPES(value.type(), "deform_conv_backward_cuda", ([&] {

            // gradient w.r.t. sampling location & attention weight
            ms_deformable_col2im_coord_cuda(at::cuda::getCurrentCUDAStream(),
                                            grad_output_g.data<scalar_t>(),
                                            value.data<scalar_t>() + n * im2col_step_ * per_value_size,
                                            spatial_shapes.data<int64_t>(),
                                            level_start_index.data<int64_t>(),
                                            sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                                            attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                                            batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                                            grad_sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                                            grad_attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size);
            // gradient w.r.t. value
            ms_deformable_col2im_cuda(at::cuda::getCurrentCUDAStream(),
                                    grad_output_g.data<scalar_t>(),
                                    spatial_shapes.data<int64_t>(),
                                    level_start_index.data<int64_t>(),
                                    sampling_loc.data<scalar_t>() + n * im2col_step_ * per_sample_loc_size,
                                    attn_weight.data<scalar_t>() + n * im2col_step_ * per_attn_weight_size,
                                    batch_n, spatial_size, num_heads, channels, num_levels, num_query, num_point,
                                    grad_value.data<scalar_t>() +  n * im2col_step_ * per_value_size);

        }));
    }

    return {
        grad_value, grad_sampling_loc, grad_attn_weight
    };
}