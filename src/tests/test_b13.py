"""Batch 13 options: defaults reproduce the published recipe exactly; the new options do what they claim (CPU)."""

import hashlib
import os
import platform
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
if 'downstream.custom' not in sys.modules:            # rr.data subclasses the starter's classes; stub them on CPU
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(SRC), 'chameleon'))
        import downstream.custom  # noqa: F401
    except Exception:
        sys.path.pop(0)
        for k in [k for k in sys.modules if k == 'downstream' or k.startswith('downstream.')]:
            del sys.modules[k]
        mc = types.ModuleType('downstream.custom')

        class _Learner:
            def compute_loss(self, Y_pred, Y, M):
                import torch.nn.functional as F
                loss = (M * F.binary_cross_entropy_with_logits(Y_pred, Y, reduction='none')).mean()
                return loss, {'loss': loss.detach()}

        mc.CustomDenseDataset, mc.CustomLearner = type('CustomDenseDataset', (), {}), _Learner
        sys.modules['downstream'] = types.ModuleType('downstream')
        sys.modules['downstream.custom'] = mc
from rr import data as D  # noqa: E402
from torchvision.transforms import ColorJitter  # noqa: E402

# md5 of FloodDataset.postprocess outputs (20 training crops + the support crop) computed with the code before
# Batch 13 (torch 2.14.1 CPU, macOS arm64: float kernels differ on other platforms); the defaults must keep giving exactly these tensors and random draws.
REF_POSTPROCESS = 'fe82fcfd2b06220f94a3717ea91a2d97'     # one CPU thread (14 Oct; = the published code, which gives 3e17d5... with the default threads)


def sample():
    g = torch.Generator().manual_seed(0)
    X = torch.rand(3, 576, 768, generator=g)
    lab = torch.randint(0, 5, (576, 768), generator=g)
    Y = torch.stack([(lab == c) for c in range(5)]).float()
    return X, Y, torch.ones_like(Y)


def dataset(**opts):
    d = D.FloodDataset.__new__(D.FloodDataset)
    d.img_size = (384, 384)
    d.jitter = ColorJitter(brightness=0.2, contrast=0.2)
    for k, v in opts.items():
        setattr(d, k, v)
    return d


def test_postprocess_defaults_bit_exact():
    if torch.__version__.split('+')[0] != '2.14.1' or (sys.platform, platform.machine()) != ('darwin', 'arm64'):
        pytest.skip('reference hash recorded with torch 2.14.1 on macOS arm64')
    X, Y, M = sample()
    h = hashlib.md5()
    d = dataset(eval_mode=False, resize=False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)               # the resampling results depend on the CPU thread count (OMP_NUM_THREADS)
    try:
        for s in range(20):
            np.random.seed(s)
            torch.manual_seed(s)
            for t in d.postprocess(X, Y, M):
                h.update(t.numpy().tobytes())
        d.eval_mode, d.resize = True, True
        for t in d.postprocess(X, Y, M):
            h.update(t.numpy().tobytes())
    finally:
        torch.set_num_threads(threads)
    assert h.hexdigest() == REF_POSTPROCESS


def test_support_views_differ_and_overlap_little():
    X, Y, M = sample()
    o0 = D.best_crop_offset(Y, M, 384)
    assert D.best_crop_offset(Y, M, 384, view=0) == o0
    o1, o2 = D.best_crop_offset(Y, M, 384, view=1), D.best_crop_offset(Y, M, 384, view=2)
    ov = lambda a, b: max(0, 384 - abs(a[0] - b[0])) * max(0, 384 - abs(a[1] - b[1])) / 384 ** 2
    assert o1 != o0 and ov(o0, o1) < 0.5
    assert ov(o0, o2) < 0.5 and ov(o1, o2) < 0.5
    w = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    assert D.best_crop_offset(Y, M, 384, weights=w) is not None


def test_rare_crop_centres_on_rare_pixels():
    X, Y, M = sample()
    Y = torch.zeros_like(Y)
    Y[0] = 1
    Y[0, 500:510, 700:710], Y[1, 500:510, 700:710] = 0, 1            # one small flooded-road patch
    d = dataset(eval_mode=False, resize=False, rare_crop_p=1.0)
    for s in range(5):
        np.random.seed(s)
        torch.manual_seed(s)
        _, Yc, _ = d.postprocess(X, Y, M)
        assert Yc[1].sum() > 0


def test_rot_blur_and_copy_paste_shapes():
    X, Y, M = sample()
    np.random.seed(1)
    torch.manual_seed(1)
    for s in range(6):
        Xr, Yr, Mr = D.FloodDataset.rotate_blur(X[:, :384, :384], Y[:, :384, :384], M[:, :384, :384])
        assert Xr.shape == (3, 384, 384) and Yr.shape == (5, 384, 384) and set(Yr.unique().tolist()) <= {0.0, 1.0}
    d = dataset(eval_mode=False, resize=False, copy_paste_p=1.0, stems=['a'])
    d.base_resizer = lambda x: x
    d.load_image = lambda stem: X.flip(-1)
    d.load_label = lambda stem: (Y.flip(-1), M)
    Xo, Yo, Mo = d.copy_paste(X, Y, M)
    assert Xo.shape == X.shape and torch.equal(Yo.sum(0), torch.ones(576, 768))


def test_loss_defaults_equal_starter_bce():
    g = torch.Generator().manual_seed(0)
    P = torch.randn(1, 5, 2, 1, 32, 32, generator=g)
    Y = (torch.rand(1, 5, 2, 1, 32, 32, generator=g) > 0.7).float()
    M = (torch.rand(1, 5, 2, 1, 32, 32, generator=g) > 0.1).float()
    import torch.nn.functional as F
    ref = (M * F.binary_cross_entropy_with_logits(P, Y, reduction='none')).mean()
    assert torch.equal(D.rr_loss(P, Y, M)[0], ref)
    assert torch.allclose(D.rr_loss(P, Y, M, pos_weight=[1] * 5)[0], ref)
    up = D.rr_loss(P, Y, M, pos_weight=[1, 3, 1, 3, 1])[0]
    assert up > ref
    ld, parts = D.rr_loss(P, Y, M, dice=0.5)
    assert torch.allclose(ld, ref + 0.5 * parts['loss_dice'])
    L = D.FloodLearner.__new__(D.FloodLearner)
    assert torch.equal(L.compute_loss(P, Y, M)[0], ref)


def test_avg_ckpt(tmp_path):
    import avg_ckpt
    for k, v in (('a', 1.0), ('b', 3.0)):
        torch.save({'state_dict': {'w': torch.full((2,), v), 'n': torch.tensor([7])}, 'hyper_parameters': {'config': k}},
                   tmp_path / f'{k}.ckpt')
    out = avg_ckpt.average([tmp_path / 'a.ckpt', tmp_path / 'b.ckpt'])
    assert torch.equal(out['state_dict']['w'], torch.full((2,), 2.0))
    assert out['state_dict']['n'].item() == 7 and out['hyper_parameters']['config'] == 'a'


def test_postproc_helpers():
    import postproc_sweep as PS
    from rr.metrics import decode
    rng = np.random.default_rng(0)
    P = rng.random((5, 40, 50)).astype(np.float32)
    cls = decode(P, tau=0.5)
    assert np.array_equal(PS.cleanup(cls, 0, 0), cls)
    assert np.array_equal(PS.decode_bias(P, [0] * 5), cls)
    I = np.full((40, 50), 0.5, np.float32)
    flat = np.full((5, 40, 50), 0.3, np.float32)
    assert np.allclose(PS.guided_filter(I, flat, 4, 1e-2), 0.3, atol=1e-5)
    two = PS.decode_two_step(P, [0.5] * 5, 0.5)
    assert two.shape == cls.shape and two.max() <= 5


def test_lora_starts_as_identity_and_trains_only_adapters():
    from rr.lora import LoRAQKV
    base = torch.nn.Linear(16, 48, bias=False)
    m = LoRAQKV(base, 4, 8)
    x = torch.randn(3, 16)
    assert torch.allclose(torch.nn.functional.linear(x, m.weight), base(x))
    assert [n for n, _ in m.named_parameters()] == ['lora_A', 'lora_B']
    m.lora_B.data.normal_()
    torch.nn.functional.linear(x, m.weight).sum().backward()
    assert m.lora_A.grad is not None and m.lora_B.grad is not None


def test_final_decode_defaults_and_chain(tmp_path):
    from rr.final_decode import decode_photo
    from rr.metrics import decode
    import postproc_sweep as PS
    rng = np.random.default_rng(1)
    P = rng.random((5, 60, 80)).astype(np.float32)
    img = (rng.random((60, 80, 3)) * 255).astype(np.uint8)
    dec = {'tau': [0.4, 0.35, 0.45, 0.35, 0.45], 'variant': 'tau'}
    assert np.array_equal(decode_photo(P, dec), decode(P, tau=dec['tau']))
    dec2 = dict(dec, gf=[2, 1e-3], clean=[50, 1])
    ref = PS.cleanup(decode(PS.guided_filter(img.mean(2).astype(np.float32) / 255, P, 2, 1e-3), tau=dec['tau']), 50, 1)
    assert np.array_equal(decode_photo(P, dec2, img=img), ref)
    np.savez(tmp_path / 'm.npz', ids=np.array(['7']), maps=ref[None])
    dec3 = dict(dec2, classmaps={'s': str(tmp_path / 'm.npz')})
    assert np.array_equal(decode_photo(P * 0, dec3, split_file='s', photo_id='7'), ref)


def test_final_decode_lazy_photo_only_when_needed():
    from rr.final_decode import decode_photo, needs_photo
    rng = np.random.default_rng(2)
    P = rng.random((5, 30, 40)).astype(np.float32)
    calls = []

    def loader():
        calls.append(1)
        return (rng.random((30, 40, 3)) * 255).astype(np.uint8)
    plain = {'tau': [0.5] * 5}
    decode_photo(P, plain, img=loader)
    assert not calls and not needs_photo(plain)               # the published decode never loads the photo
    gf = {'tau': [0.5] * 5, 'gf': [2, 1e-3]}
    decode_photo(P, gf, img=loader)
    assert calls == [1] and needs_photo(gf)


def test_run_name_support_seed():
    from rr.runs import run_name, support_seed_of
    assert run_name('floodnet', 50, 0) == 'floodnet_50shot_s0_stratified'            # published names unchanged
    assert run_name('floodnet', 50, 1, support_seed=1) == 'floodnet_50shot_s1_stratified'
    assert run_name('floodnet', 50, 1, support_seed=2, tag='p2') == 'floodnet_50shot_s1_stratified_sup2_p2'
    assert support_seed_of(types.SimpleNamespace(seed=3)) == 3
    assert support_seed_of(types.SimpleNamespace(seed=3, rr_support_seed=1)) == 1
