#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch


N, M, D = 2, 2, 4
Lq, L, P = 3, 3, 2
shapes = torch.as_tensor([(8, 8), (4, 4), (2, 2)], dtype=torch.long).cuda()
S = sum([(H*W).item() for H, W in shapes])


torch.manual_seed(3)


def check_forward_equal_with_pytorch():
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    output_pytorch = ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attention_weights)
    output_cuda = MSDeformAttnFunction.apply(value, shapes, sampling_locations, attention_weights, im2col_step)
    fwdok = torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()

    print(f'* {fwdok} check_forward_equal_with_pytorch: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


def check_backward_equal_with_pytorch():
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    value.requires_grad = True
    sampling_locations.requires_grad = True
    attention_weights.requires_grad = True
    output_pytorch = ms_deform_attn_core_pytorch(value, shapes, sampling_locations, attention_weights)
    output_cuda = MSDeformAttnFunction.apply(value, shapes, sampling_locations, attention_weights, im2col_step)
    loss_pytorch = output_pytorch.abs().sum()
    loss_cuda = output_cuda.abs().sum()

    grad_value_pytorch = torch.autograd.grad(loss_pytorch, value, retain_graph=True)[0]
    grad_value_cuda = torch.autograd.grad(loss_cuda, value, retain_graph=True)[0]
    bwdok = torch.allclose(grad_value_cuda, grad_value_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (grad_value_cuda - grad_value_pytorch).abs().max()
    zero_grad_mask = grad_value_pytorch == 0
    max_rel_err = ((grad_value_cuda - grad_value_pytorch).abs() / grad_value_pytorch.abs())[~zero_grad_mask].max()
    if zero_grad_mask.sum() == 0:
        max_abs_err_0 = 0
    else:
        max_abs_err_0 = (grad_value_cuda - grad_value_pytorch).abs()[zero_grad_mask].max()
    print(f'* {bwdok} check_backward_equal_with_pytorch - input1: '
          f'max_abs_err {max_abs_err:.2e} '
          f'max_rel_err {max_rel_err:.2e} '
          f'max_abs_err_0 {max_abs_err_0:.2e}')

    grad_sampling_loc_pytorch = torch.autograd.grad(loss_pytorch, sampling_locations, retain_graph=True)[0]
    grad_sampling_loc_cuda = torch.autograd.grad(loss_cuda, sampling_locations, retain_graph=True)[0]
    bwdok = torch.allclose(grad_sampling_loc_cuda, grad_sampling_loc_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (grad_sampling_loc_cuda - grad_sampling_loc_pytorch).abs().max()
    zero_grad_mask = grad_sampling_loc_pytorch == 0
    max_rel_err = ((grad_sampling_loc_cuda - grad_sampling_loc_pytorch).abs() / grad_sampling_loc_pytorch.abs())[~zero_grad_mask].max()
    if zero_grad_mask.sum() == 0:
        max_abs_err_0 = 0
    else:
        max_abs_err_0 = (grad_sampling_loc_cuda - grad_sampling_loc_pytorch).abs()[zero_grad_mask].max()
    print(f'* {bwdok} check_backward_equal_with_pytorch - input2: '
          f'max_abs_err {max_abs_err:.2e} '
          f'max_rel_err {max_rel_err:.2e} '
          f'max_abs_err_0 {max_abs_err_0:.2e}')

    grad_attn_weight_pytorch = torch.autograd.grad(loss_pytorch, attention_weights, retain_graph=True)[0]
    grad_attn_weight_cuda = torch.autograd.grad(loss_cuda, attention_weights, retain_graph=True)[0]
    bwdok = torch.allclose(grad_attn_weight_cuda, grad_attn_weight_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (grad_attn_weight_cuda - grad_attn_weight_pytorch).abs().max()
    zero_grad_mask = grad_attn_weight_pytorch == 0
    max_rel_err = ((grad_attn_weight_cuda - grad_attn_weight_pytorch).abs() / grad_attn_weight_pytorch.abs())[~zero_grad_mask].max()
    if zero_grad_mask.sum() == 0:
        max_abs_err_0 = 0
    else:
        max_abs_err_0 = (grad_attn_weight_cuda - grad_attn_weight_pytorch).abs()[zero_grad_mask].max()
    print(f'* {bwdok} check_backward_equal_with_pytorch - input3: '
          f'max_abs_err {max_abs_err:.2e} '
          f'max_rel_err {max_rel_err:.2e} '
          f'max_abs_err_0 {max_abs_err_0:.2e}')


def check_gradient_ms_deform_attn(
        use_pytorch=False,
        grad_value=True, grad_sampling_loc=True, grad_attn_weight=True):

    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    if use_pytorch:
        func = ms_deform_attn_core_pytorch
    else:
        func = MSDeformAttnFunction.apply

    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    attention_weights.requires_grad = grad_attn_weight

    eps = 1e-3 if not grad_sampling_loc else 2e-4
    if use_pytorch:
        gradok = gradcheck(func, (value, shapes, sampling_locations, attention_weights),
                           eps=eps, atol=1e-3, rtol=1e-2, raise_exception=True)
    else:
        gradok = gradcheck(func, (value, shapes, sampling_locations, attention_weights, im2col_step),
                           eps=eps, atol=1e-3, rtol=1e-2, raise_exception=True)

    print(f'* {gradok} '
          f'check_gradient_ms_deform_attn('
          f'{use_pytorch}, {grad_value}, {grad_sampling_loc}, {grad_attn_weight})')


if __name__ == '__main__':
    print('checking forward')
    check_forward_equal_with_pytorch()

    print('checking backward')
    check_backward_equal_with_pytorch()

    print('checking gradient of pytorch version')
    check_gradient_ms_deform_attn(True, True, False, False)
    check_gradient_ms_deform_attn(True, False, True, False)
    check_gradient_ms_deform_attn(True, False, False, True)
    check_gradient_ms_deform_attn(True, True, True, True)

    print('checking gradient of cuda version')
    check_gradient_ms_deform_attn(False, True, False, False)
    check_gradient_ms_deform_attn(False, False, True, False)
    check_gradient_ms_deform_attn(False, False, False, True)
    check_gradient_ms_deform_attn(False, True, True, True)



