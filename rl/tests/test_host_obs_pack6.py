"""GPU-only lossless codec and asynchronous rollout/gather boundary checks."""
import ctypes
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlenv import _load, build_cuda
from train import HostObs
from model import Policy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires the owned GPU')


def test_codec_all_colours_and_random_quads():
    lib = _load(build_cuda())
    gen = torch.Generator(device='cuda').manual_seed(29)
    raw = torch.randint(0, 64, (131072, 4), dtype=torch.uint8, device='cuda', generator=gen)
    raw[:64] = torch.arange(64, dtype=torch.uint8, device='cuda')[:, None]
    packed = torch.empty(raw.numel() * 3 // 4, dtype=torch.uint8, device='cuda')
    decoded = torch.empty_like(raw)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    assert lib.rl_pack6(ctypes.c_void_p(raw.data_ptr()), ctypes.c_void_p(packed.data_ptr()), raw.numel(), stream) == 0
    assert lib.rl_unpack6(ctypes.c_void_p(packed.data_ptr()), ctypes.c_void_p(decoded.data_ptr()), raw.numel(), stream) == 0
    assert torch.equal(decoded, raw)
    x = raw.long()
    words = x[:, 0] | (x[:, 1] << 6) | (x[:, 2] << 12) | (x[:, 3] << 18)
    reference = torch.stack([words & 255, (words >> 8) & 255, (words >> 16) & 255], 1).byte().flatten()
    assert torch.equal(packed, reference)


@pytest.mark.parametrize('nslots', [1, 2])
@pytest.mark.parametrize('per_unit', [False, True])
def test_shuffled_gather_ping_pong_and_rollout_handoff(nslots, per_unit):
    T, N, mb = 6, 12, 3
    lib = _load(build_cuda())
    store = HostObs(lib, T, N, torch.device('cuda'), mb, 2, nslots, pack6=True)
    gen = torch.Generator(device='cuda').manual_seed(32)
    references = []
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for region in range(2):
        frames = torch.randint(0, 64, (T+1, N, 4, 224, 240), dtype=torch.uint8, device='cuda', generator=gen)
        if references:
            frames[0].copy_(references[-1][-1])  # real T→0 handoff
        references.append(frames)
        main = torch.cuda.current_stream()
        for stream in streams:
            stream.wait_stream(main)
        for t in range(T+1):
            if per_unit:
                for gi, rows in enumerate((slice(0, 5), slice(5, N))):
                    with torch.cuda.stream(streams[gi]):
                        store.before_write_game(gi, t-1)
                        store.slot(region, t)[rows].copy_(frames[t, rows])
                        store.saved_game(region, t, gi, rows)
            else:
                store.before_write(t-1)
                store.slot(region, t).copy_(frames[t])
                store.saved(region, t)
        for stream in streams:
            main.wait_stream(stream)
        torch.cuda.synchronize()
    assert store.region_bytes == (T+1)*N*4*224*240*3//4
    for region in range(2):
        for _ in range(2):
            perm = torch.randperm(N, device='cuda', generator=gen)
            store.epoch(region, perm, mb)
            for i in range(N//mb):
                actual = store.batch(i)
                expected = references[region][:T, perm[i*mb:(i+1)*mb]].flatten(0, 1)
                assert torch.equal(actual, expected), (region, i)
                store.done_with(i)
    torch.cuda.synchronize()


def test_decoded_inputs_preserve_values_logprobs_gradients_and_adam_step():
    import copy
    torch.manual_seed(91)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    lib = _load(build_cuda())
    T, N = 128, 1
    raw = torch.randint(0, 64, (T*N, 4, 224, 240), dtype=torch.uint8, device='cuda')
    encoded = torch.empty(raw.numel()*3//4, dtype=torch.uint8, device='cuda')
    decoded = torch.empty_like(raw)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    assert lib.rl_pack6(ctypes.c_void_p(raw.data_ptr()), ctypes.c_void_p(encoded.data_ptr()), raw.numel(), stream) == 0
    assert lib.rl_unpack6(ctypes.c_void_p(encoded.data_ptr()), ctypes.c_void_p(decoded.data_ptr()), raw.numel(), stream) == 0
    emph = torch.randint(0, 8, (T*N, 4, 224), dtype=torch.uint8, device='cuda')
    timeline = torch.zeros(T*N, 8, 45, dtype=torch.uint8, device='cuda')
    previous = [torch.zeros(T*N, dtype=torch.long, device='cuda') for _ in range(5)]
    masks = torch.ones(T*N, 19, dtype=torch.uint8, device='cuda')
    initial = Policy(core=64).cuda().state_dict()

    def update(images):
        model = Policy(core=64).cuda()
        model.load_state_dict(initial)
        opt = torch.optim.Adam(model.parameters(), lr=2.5e-4)
        z = model.encode(images, emph, timeline, previous)
        starts = torch.zeros(T, N, dtype=torch.bool, device='cuda')
        starts[0] = True
        h, _ = model.recur(z.view(T, N, -1), torch.zeros(N, model.core, device='cuda'), starts)
        h = h.flatten(0, 1)
        _, logp = model.act(h, masks, greedy=True)
        value = model.value(h)
        loss = -logp.mean() + value.square().mean()
        loss.backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        assert all(torch.isfinite(g).all() for g in gradients)
        opt.step()
        return logp.detach(), value.detach(), gradients, copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())

    a, b = update(raw), update(decoded)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert all(torch.equal(x, y) for x, y in zip(a[2], b[2]))
    assert all(torch.equal(a[3][k], b[3][k]) for k in a[3])
    for key, state in a[4]['state'].items():
        for name, x in state.items():
            y = b[4]['state'][key][name]
            assert torch.equal(x, y) if torch.is_tensor(x) else x == y
