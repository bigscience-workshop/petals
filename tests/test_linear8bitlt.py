import bitsandbytes as bnb
import pytest
import torch
from bitsandbytes import functional as F

from petals.utils.linear8bitlt_patch import CustomLinear8bitLt, get_inverse_transform_indices, logger, undo_layout


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 5),
    reason="this test requires a turing-generation or newer GPU, see bitsandbytes docs",
)
def test_layout_exact_match():
    logger.warning("Testing inverse CxB transform")
    x = (torch.randn(14336 * 3, 14336) * 10).to(torch.int8).cuda()
    for tile_size, order in ((8, 32), "col_turing"), ((32, 32), "col_ampere"):
        transform = lambda x: F.transform(x.cuda(), from_order="row", to_order=order)[0].to(x.device)
        tile_indices = get_inverse_transform_indices(transform, tile_size)
        cxb = transform(x)

        torch.cuda.synchronize()
        restored_x = undo_layout(cxb, tile_indices)
        torch.cuda.synchronize()
        assert restored_x.is_contiguous()
        assert torch.all(torch.eq(restored_x, x))
    logger.warning("CxB tests passed")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 5),
    reason="this test requires a turing-generation or newer GPU, see bitsandbytes docs",
)
def test_linear_exact_match():
    linear = torch.nn.Linear(1024, 3072)
    x = torch.randn(3, 1024, dtype=torch.half)
    linear8bitlt = bnb.nn.Linear8bitLt(
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        has_fp16_weights=False,
        threshold=6.0,
        memory_efficient_backward=True,
    )
    linear8bitlt.weight = bnb.nn.Int8Params(linear.weight.data, requires_grad=False, has_fp16_weights=False).to(
        linear.weight.dtype
    )
    linear8bitlt.cuda()

    linear_custom = CustomLinear8bitLt(
        linear.in_features,
        linear.out_features,
        linear.bias is not None,
        has_fp16_weights=False,
        threshold=6.0,
    )
    linear_custom.weight = bnb.nn.Int8Params(linear.weight.data, requires_grad=False, has_fp16_weights=False).to(
        linear.weight.dtype
    )
    linear8bitlt.cuda()

    x_ref = x.clone().cuda().requires_grad_(True)
    x_ours = x.clone().cuda().requires_grad_(True)
    fx_ref = linear8bitlt(x_ref).float()
    grad_proj = torch.randn_like(fx_ref)
    (fx_ref * grad_proj).mean().backward()

    fx_ours = linear8bitlt(x_ours).float()
    (fx_ours * grad_proj).mean().backward()
    assert torch.equal(fx_ref, fx_ours)
    assert torch.allclose(x_ref.grad, x_ours.grad)
