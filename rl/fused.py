"""Fast colour lookup for the model input (CUDA): NES colour numbers + emphasis of 4 frames -> channels-last
[B, 224, 240, 16] picture, 4 frames x (R, G, B, 0). One thread block handles a run of pixels and writes each pixel's
16 values contiguously; the compiler's generic gather read the frame planes in a scattered order (19 ms per 2048
samples on the RTX 5090). Values are exactly the palette entries (no arithmetic), so it equals Policy.rgb(pad=True)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rgb16_kernel(obs_ptr, emph_ptr, pal_ptr, out_ptr, n_pix, HW, H, W, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = p < n_pix
    b = p // HW
    r = p % HW
    h = r // W
    c = tl.arange(0, 16)
    f = c // 4
    ch = c % 4
    plane = b[:, None] * 4 + f[None, :]
    o = tl.load(obs_ptr + plane * HW + r[:, None], mask=m[:, None], other=0).to(tl.int32)
    e = tl.load(emph_ptr + plane * H + h[:, None], mask=m[:, None], other=0).to(tl.int32)
    v = tl.load(pal_ptr + (o | (e << 6)) * 4 + ch[None, :])
    tl.store(out_ptr + p[:, None] * 16 + c[None, :], v, mask=m[:, None])


@torch.library.custom_op('rl::rgb16', mutates_args=())
def rgb16(obs: torch.Tensor, emph: torch.Tensor, pal4: torch.Tensor) -> torch.Tensor:
    B, F_, H, W = obs.shape
    assert F_ == 4 and obs.is_contiguous() and emph.is_contiguous() and pal4.shape == (512, 4)
    out = torch.empty(B, H, W, 16, dtype=pal4.dtype, device=obs.device)
    n = B * H * W
    BLOCK = 256
    _rgb16_kernel[(triton.cdiv(n, BLOCK),)](obs, emph, pal4.contiguous(), out, n, H * W, H, W, BLOCK=BLOCK)
    return out


@rgb16.register_fake
def _(obs, emph, pal4):
    B, F_, H, W = obs.shape
    return obs.new_empty(B, H, W, 16, dtype=pal4.dtype)


# ---------------------------------------------------------------------------------------------------------------
# Fused colour lookup + stem convolution (2x2, stride 2, 12 -> 16 channels). The stem reads each input pixel once, so
# its output can be computed straight from the colour numbers: out[p, o] = bias[o] + sum_k W[o, k] * x[p, k] over the
# 48 inputs k = (frame, R/G/B, dy, dx) of output pixel p, x = the palette value of that pixel's colour. The backward
# pass needs only dW = sum_p g[p] x[p]^T and dbias = sum_p g[p] (the input is not trained), and recomputes x from the
# colour numbers instead of keeping the 1.7 MB/sample picture. Same products as the bf16 convolution (bf16 inputs and
# weights, fp32 sums, on tensor cores); only the order of the fp32 additions differs. bf16 autocast only.

@triton.jit
def _stem_inputs(obs_ptr, emph_ptr, pal_ptr, p, m, OW, OHW, H, W, BLOCK: tl.constexpr):
    """x[p, k] for a block of output pixels: k = (frame * 3 + RGB) * 4 + dy * 2 + dx (48 used of 64), bf16 values."""
    b = p // OHW
    r = p % OHW
    oy = r // OW
    ox = r % OW
    k = tl.arange(0, 64)
    f = k // 12
    ch = (k // 4) % 3
    dy = (k // 2) % 2
    dx = k % 2
    mk = m[:, None] & (k < 48)[None, :]
    plane = b[:, None] * 4 + f[None, :]
    y = oy[:, None] * 2 + dy[None, :]
    x = ox[:, None] * 2 + dx[None, :]
    c = tl.load(obs_ptr + (plane * H + y) * W + x, mask=mk, other=0).to(tl.int32)
    e = tl.load(emph_ptr + plane * H + y, mask=mk, other=0).to(tl.int32)
    v = tl.load(pal_ptr + (c | (e << 6)) * 3 + ch[None, :], mask=mk, other=0.0)
    return v.to(tl.bfloat16)


@triton.jit
def _stem_fwd_kernel(obs_ptr, emph_ptr, pal_ptr, wt_ptr, b_ptr, out_ptr, n_out, OW, OHW, H, W,
                     BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = p < n_out
    xk = _stem_inputs(obs_ptr, emph_ptr, pal_ptr, p, m, OW, OHW, H, W, BLOCK)       # [BLOCK, 64] bf16
    k = tl.arange(0, 64)
    o = tl.arange(0, 16)
    wt = tl.load(wt_ptr + k[:, None] * 16 + o[None, :])                            # [64, 16] bf16 (rows >= 48 zero)
    acc = tl.dot(xk, wt) + tl.load(b_ptr + o).to(tl.float32)[None, :]
    tl.store(out_ptr + p[:, None] * 16 + o[None, :], acc.to(out_ptr.dtype.element_ty), mask=m[:, None])


@triton.jit
def _stem_bwd_kernel(g_ptr, obs_ptr, emph_ptr, pal_ptr, dw_ptr, db_ptr, n_out, per_prog, OW, OHW, H, W,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    o = tl.arange(0, 16)
    k = tl.arange(0, 64)
    acc = tl.zeros((64, 16), dtype=tl.float32)
    accb = tl.zeros((16,), dtype=tl.float32)
    start = pid * per_prog
    for s in range(0, per_prog, BLOCK):
        p = start + s + tl.arange(0, BLOCK)
        m = (p < n_out) & (s + tl.arange(0, BLOCK) < per_prog)
        g = tl.load(g_ptr + p[:, None] * 16 + o[None, :], mask=m[:, None], other=0)  # [BLOCK, 16] bf16
        accb += tl.sum(g.to(tl.float32), axis=0)
        xk = _stem_inputs(obs_ptr, emph_ptr, pal_ptr, p, m, OW, OHW, H, W, BLOCK)
        acc += tl.dot(tl.trans(xk), g.to(tl.bfloat16))
    tl.store(dw_ptr + pid * 64 * 16 + k[:, None] * 16 + o[None, :], acc)
    tl.store(db_ptr + pid * 16 + o, accb)


@torch.library.custom_op('rl::stem_fwd', mutates_args=())
def stem_fwd(obs: torch.Tensor, emph: torch.Tensor, pal: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    B, F_, H, W = obs.shape
    OH, OW = H // 2, W // 2
    out = torch.empty(B, OH, OW, 16, dtype=w.dtype, device=obs.device)
    n = B * OH * OW
    BLOCK = 128
    wt = torch.zeros(64, 16, dtype=torch.bfloat16, device=obs.device)
    wt[:48] = w.t().to(torch.bfloat16)
    _stem_fwd_kernel[(triton.cdiv(n, BLOCK),)](obs, emph, pal, wt, b.contiguous(), out, n, OW, OH * OW,
                                               H, W, BLOCK=BLOCK)
    return out


@stem_fwd.register_fake
def _(obs, emph, pal, w, b):
    B, F_, H, W = obs.shape
    return obs.new_empty(B, H // 2, W // 2, 16, dtype=w.dtype)


@torch.library.custom_op('rl::stem_bwd', mutates_args=())
def stem_bwd(g: torch.Tensor, obs: torch.Tensor, emph: torch.Tensor, pal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, F_, H, W = obs.shape
    OH, OW = H // 2, W // 2
    n = B * OH * OW
    per = 16384                                   # pixels per program (swept on the RTX 5090)
    nprog = triton.cdiv(n, per)
    dw = torch.empty(nprog, 64, 16, dtype=torch.float32, device=obs.device)
    db = torch.empty(nprog, 16, dtype=torch.float32, device=obs.device)
    _stem_bwd_kernel[(nprog,)](g.to(torch.bfloat16).contiguous(), obs, emph, pal, dw, db, n, per, OW, OH * OW, H, W,
                               BLOCK=64, num_warps=8)
    return dw.sum(0)[:48].t().contiguous(), db.sum(0)            # dW [16, 48], dbias [16]


@stem_bwd.register_fake
def _(g, obs, emph, pal):
    return g.new_empty(16, 48, dtype=torch.float32), g.new_empty(16, dtype=torch.float32)


def _stem_setup(ctx, inputs, output):
    obs, emph, pal, w, b = inputs
    ctx.save_for_backward(obs, emph, pal)
    ctx.wdtype, ctx.bdtype = w.dtype, b.dtype


def _stem_backward(ctx, grad):
    obs, emph, pal = ctx.saved_tensors
    dw, db = stem_bwd(grad, obs, emph, pal)
    return None, None, None, dw.to(ctx.wdtype), db.to(ctx.bdtype)


stem_fwd.register_autograd(_stem_backward, setup_context=_stem_setup)


def fused_stem(obs, emph, palette_rgb, weight, bias, dtype):
    """obs [B,4,224,240] uint8, emph [B,4,224] uint8, palette_rgb [512,3] fp32, weight [16,12,2,2], bias [16] ->
    stem output [B,16,112,120] in `dtype`, channels-last strides."""
    pal = palette_rgb.to(dtype).float().contiguous()                     # the bf16 picture values
    w = weight.reshape(weight.shape[0], 48).to(dtype)
    out = stem_fwd(obs.contiguous(), emph.contiguous(), pal, w, bias.to(dtype))
    return out.permute(0, 3, 1, 2)
