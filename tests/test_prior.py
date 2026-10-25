import math

import torch

from jem.config import make_config
from jem.data import Window
from jem.energy import build_model
from jem.inference import Proposals, infer_newton, newton_solve
from jem.objectives import candidates, jem_loss
from jem.prior import BehaviourPrior


def _prior(da=2, seed=0):
    torch.manual_seed(seed)
    return BehaviourPrior(3, da, -1.0, 1.0, components=4, hidden=32).double()


def test_prior_is_a_normalised_density_and_samples_follow_it():
    prior = _prior()
    obs = torch.randn(3, 3, dtype=torch.float64)
    x = torch.linspace(-1, 1, 801, dtype=torch.float64)
    grid = torch.stack(torch.meshgrid(x, x, indexing="ij"), -1).reshape(-1, 2)
    dx = (x[1] - x[0]).item()
    p = prior.log_prob(obs, grid.expand(3, *grid.shape)).exp()  # (3, G)
    mass = p.sum(-1) * dx * dx  # Riemann sum; the density is smooth on the box
    assert torch.allclose(mass, torch.ones(3, dtype=torch.float64), atol=5e-3)
    a = prior.sample(obs, 50_000, torch.Generator().manual_seed(1))
    assert a.shape == (3, 50_000, 2) and a.abs().max() <= 1
    mean = (p[..., None] * grid).sum(1) * dx * dx
    assert torch.allclose(a.mean(1), mean, atol=1e-2)


def test_mixture_density_bounds_the_offsets():
    """r = (1 - rho) q + rho U >= rho U, so the offsets -log r are bounded above."""
    prior = _prior()
    obs = torch.randn(5, 3, dtype=torch.float64)
    act = torch.rand(5, 64, 2, dtype=torch.float64) * 2 - 1
    lr = prior.log_mixture(obs, act, 0.25)
    assert (lr >= math.log(0.25) - prior.log_volume - 1e-9).all()
    lq = prior.log_prob(obs, act)
    assert torch.allclose(lr, torch.log(0.75 * lq.exp() + 0.25 / 4.0))


def test_offsets_enter_newton_exactly_and_constant_offsets_do_nothing():
    g = torch.Generator().manual_seed(0)
    U = 2.0 * torch.randn(4, 5, 33, 3, generator=g, dtype=torch.float64)
    offs = torch.randn(4, 5, 33, generator=g, dtype=torch.float64)
    z0, _ = newton_solve(U, 1.0, 30)
    zc, _ = newton_solve(U, 1.0, 30, offs=torch.full_like(offs, 3.0))
    assert torch.allclose(z0, zc)
    z, res = newton_solve(U, 1.0, 30, offs=offs)
    assert res.max() < 1e-8 and not torch.allclose(z, z0)
    # The implicit gradient with offsets matches finite differences.
    U0 = 2.0 * torch.randn(4, 5, 33, 3, generator=g, dtype=torch.float64)
    w = torch.randn(4, 3, generator=g, dtype=torch.float64)
    theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    zt, _ = infer_newton(theta * U0 + U, 1.0, 30, True, None, offs)
    (grad,) = torch.autograd.grad((w * zt).sum(), theta)
    eps = 1e-4
    zp, _ = newton_solve((0.7 + eps) * U0 + U, 1.0, 30, offs=offs)
    zm, _ = newton_solve((0.7 - eps) * U0 + U, 1.0, 30, offs=offs)
    assert torch.allclose(grad, ((w * zp).sum() - (w * zm).sum()) / (2 * eps), rtol=1e-5)


def test_jem_loss_with_a_prior_trains_and_orders_candidates():
    cfg = make_config("jem", ["objective.proposal=prior", "objective.num_proposals=32"])
    torch.manual_seed(0)
    model = build_model(cfg.model, 4, 2)
    prior = BehaviourPrior(4, 2, -1.0, 1.0, components=4, hidden=32)
    win = Window(torch.randn(6, 16, 4), torch.rand(6, 16, 2) * 2 - 1, torch.zeros(6, 16))
    props = Proposals(2, -1.0, 1.0, 32, "cpu").sample()
    phi, offs = candidates(model, win, props, cfg, prior)
    assert phi.shape == (6, 16, 33, cfg.model.option_dim) and offs.shape == (6, 16, 33)
    # Index 0 is the observed action: its features match the model's own.
    assert torch.allclose(phi[..., 0, :], model.features(win.obs, win.act), atol=1e-6)
    loss, stats = jem_loss(model, win, props, cfg, prior)
    loss.backward()
    assert loss.isfinite() and all(v.isfinite() for v in stats.values())
    assert all(p.grad is None for p in prior.parameters())  # the prior trains on its own loss


def test_shared_candidates_match_per_state_candidates():
    cfg = make_config("jem")
    torch.manual_seed(1)
    obs, act = torch.randn(3, 5, 4), torch.rand(3, 5, 2) * 2 - 1
    own = torch.rand(3, 7, 2) * 2 - 1
    props = torch.rand(9, 2) * 2 - 1
    for arch in ("factored", "joint"):
        cfg.model.arch = arch
        model = build_model(cfg.model, 4, 2)
        shared = model.candidate_features(obs, act, props, own)
        per_state = model.candidate_features(obs, act, props, own[:, None].expand(3, 5, 7, 2))
        assert shared.shape == (3, 5, 17, cfg.model.option_dim)
        assert torch.allclose(shared, per_state, atol=1e-6)


def test_prototype_mixture_loss_trains_the_prototypes():
    cfg = make_config("jem", ["model.prototypes=3", "objective.num_proposals=32"])
    torch.manual_seed(0)
    model = build_model(cfg.model, 4, 2)
    with torch.no_grad():
        model.prototypes.normal_()
    win = Window(torch.randn(6, 16, 4), torch.rand(6, 16, 2) * 2 - 1, torch.zeros(6, 16))
    props = Proposals(2, -1.0, 1.0, 32, "cpu").sample()
    loss, stats = jem_loss(model, win, props, cfg, prototypes=True)
    loss.backward()
    assert model.prototypes.grad is not None and model.prototypes.grad.abs().sum() > 0
    assert 0 <= stats["diag/proto_entropy"] <= math.log(3) + 1e-6
    plain, _ = jem_loss(model, win, props, cfg, prototypes=False)
    assert stats["train/proto_nll"] > 0 and torch.isfinite(plain)
