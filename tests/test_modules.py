import torch

from mmod_qaf.modules import ConcatFusion, LightweightPyramidEncoder, QualityAwareFusion, estimate_input_quality


def _fusion_inputs():
    torch.manual_seed(7)
    rgb = torch.randn(2, 8, 8, 8)
    infrared = torch.randn(2, 4, 8, 8)
    depth = torch.randn(2, 4, 8, 8)
    mask = torch.ones(2, 1, 64, 64)
    quality = torch.randn(2, 7)
    return rgb, infrared, depth, mask, quality


def _fusion_blocks():
    return (ConcatFusion(8, 4, 4), QualityAwareFusion(8, 4, 4))


def test_encoder_and_qaf_shapes_and_gradients():
    x = torch.rand(2, 6, 256, 256, requires_grad=False)
    x[:, 5] = (x[:, 5] > 0.25).float()
    ir_encoder = LightweightPyramidEncoder(1, (128, 256, 512), 0.25)
    depth_encoder = LightweightPyramidEncoder(2, (128, 256, 512), 0.25)
    ir = ir_encoder(x[:, 3:4])
    depth = depth_encoder(x[:, 4:6])
    rgb = [torch.rand(2, c, 256 // stride, 256 // stride, requires_grad=True) for c, stride in zip((128,256,512),(8,16,32))]
    quality = estimate_input_quality(x)
    outputs = []
    for r, i, d, c in zip(rgb, ir, depth, (128,256,512)):
        fusion = QualityAwareFusion(c, i.shape[1], d.shape[1])
        y = fusion(r, i, d, x[:, 5:6], quality)
        assert y.shape == r.shape
        assert fusion.last_diagnostics.mean_modality_weights.shape == (2, 3)
        outputs.append(y.mean())
    sum(outputs).backward()
    assert all(r.grad is not None and torch.isfinite(r.grad).all() for r in rgb)


def test_depth_hard_mask_removes_depth_weight():
    b, c, h, w = 2, 64, 16, 16
    fusion = QualityAwareFusion(c, 16, 16).eval()
    rgb = torch.randn(b,c,h,w); ir = torch.randn(b,16,h,w); depth = torch.randn(b,16,h,w)
    mask = torch.zeros(b,1,128,128)
    with torch.no_grad(): fusion(rgb,ir,depth,mask,torch.zeros(b,7))
    weights = fusion.last_diagnostics.channel_modality_weights
    assert torch.allclose(weights[:,2], torch.zeros_like(weights[:,2]), atol=1e-7)


def test_fusion_initialization_is_exact_rgb_identity_in_train_and_eval():
    rgb, infrared, depth, mask, quality = _fusion_inputs()
    for block in _fusion_blocks():
        for training in (True, False):
            block.train(training)
            out = block(rgb, infrared, depth, mask, quality)
            torch.testing.assert_close(out, rgb, rtol=0, atol=0)


def test_fusion_auxiliary_gradients_connect_after_two_optimizer_steps():
    rgb, infrared, depth, mask, quality = _fusion_inputs()
    for block in _fusion_blocks():
        block.train()
        optimizer = torch.optim.SGD(block.parameters(), lr=0.05)

        # Build a deterministic target corresponding to the fully-open original fusion operator, then restore the
        # exact-identity initialization. This guarantees that the residual gain receives an opening gradient.
        with torch.no_grad():
            block.residual_gain.fill_(1.0)
            target = block(rgb, infrared, depth, mask, quality).detach()
            block.residual_gain.zero_()

        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = (block(rgb, infrared, depth, mask, quality) - target).square().mean()
            loss.backward()
            if step == 0:
                assert block.residual_gain.grad is not None
                assert float(block.residual_gain.grad.abs().sum()) > 0
                ir_projection_grad = sum(
                    float(p.grad.abs().sum()) for p in block.ir_proj.parameters() if p.grad is not None
                )
                depth_projection_grad = sum(
                    float(p.grad.abs().sum()) for p in block.depth_proj.parameters() if p.grad is not None
                )
                assert ir_projection_grad == 0
                assert depth_projection_grad == 0
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        infrared_probe = infrared.detach().clone().requires_grad_(True)
        depth_probe = depth.detach().clone().requires_grad_(True)
        loss = (block(rgb, infrared_probe, depth_probe, mask, quality) - target).square().mean()
        loss.backward()

        assert infrared_probe.grad is not None and float(infrared_probe.grad.abs().sum()) > 0
        assert depth_probe.grad is not None and float(depth_probe.grad.abs().sum()) > 0
        fusion_grad = sum(float(p.grad.abs().sum()) for p in block.parameters() if p.grad is not None)
        assert fusion_grad > 0
        core_fusion_grad = block.mix.weight.grad if isinstance(block, ConcatFusion) else block.gate[-1].weight.grad
        assert core_fusion_grad is not None and float(core_fusion_grad.abs().sum()) > 0
