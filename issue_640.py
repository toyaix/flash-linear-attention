"""
Minimal reproduction for https://github.com/fla-org/flash-linear-attention/issues/640
[Bug] GDN precision issue on H20 + Triton 3.5

bug in: https://github.com/fla-org/flash-linear-attention/commit/4926edd
"""

import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def recurrent_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g])
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale
    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', b_q, h)
    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h


# Parameters matching the failing test case in the original issue
B, T, H, D = 1, 1024, 4, 100
scale = 1.0
gate_logit_normalizer = 0.1
mask_p = 0.0
use_qk_l2norm_in_kernel = False
dtype = torch.float16
device = 'cuda'

torch.manual_seed(42)

# Build inputs
q = torch.rand(B, T, H, D, dtype=dtype)
k = torch.rand(B, T, H, D, dtype=dtype)
v = torch.rand(B, T, H, D, dtype=dtype)
beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32))
g = g / gate_logit_normalizer
g = g * (torch.rand_like(g) > mask_p)
h0 = torch.zeros(B, H, D, D, dtype=torch.float32)

q, k, v, beta, g, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, g, h0))

do = torch.randn(B, T, H, D, dtype=dtype, device=device)
dht = torch.randn(B, H, D, D, dtype=torch.float32, device=device)

# Triton kernel forward + backward
tri, tri_ht = chunk_gated_delta_rule(
    q=F.normalize(q.clone(), p=2, dim=-1),
    k=F.normalize(k.clone(), p=2, dim=-1),
    v=v.clone(),
    g=g.clone(),
    beta=beta.clone(),
    scale=scale,
    initial_state=h0.clone(),
    output_final_state=True,
    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
)
((tri * do).sum() + (tri_ht * dht).sum()).backward()
tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dg, tri_dh0 = \
    q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad
q.grad = k.grad = v.grad = beta.grad = g.grad = h0.grad = None

# Reference forward + backward
ref, ref_ht = recurrent_gated_delta_rule_ref(
    q=F.normalize(q.clone(), p=2, dim=-1),
    k=F.normalize(k.clone(), p=2, dim=-1),
    v=v.clone(),
    beta=beta.clone(),
    g=g.clone(),
    scale=scale,
    output_final_state=True,
    initial_state=h0.clone(),
)
((ref * do).sum() + (ref_ht * dht).sum()).backward()
ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg, ref_dh0 = \
    q.grad, k.grad, v.grad, beta.grad, g.grad, h0.grad


def assert_close(name, ref, tri, atol):
    diff = (ref.float() - tri.float()).abs()
    print(f"{name:6s}  max={diff.max():.6f}  mean={diff.mean():.6f}  atol={atol}  {'✓' if diff.max() < atol else '✗ FAIL'}")


assert_close('o',    ref,       tri,       0.005)
assert_close('ht',   ref_ht,    tri_ht,    0.005)
assert_close('dq',   ref_dq,    tri_dq,    0.008)
assert_close('dk',   ref_dk,    tri_dk,    0.008)
assert_close('dv',   ref_dv,    tri_dv,    0.008)
assert_close('db',   ref_dbeta, tri_dbeta, 0.02)   # fails on H20/H100 + Triton 3.4
assert_close('dg',   ref_dg,    tri_dg,    0.02)   # fails on H20/H100 + Triton 3.4
assert_close('dh0',  ref_dh0,   tri_dh0,   0.02)
