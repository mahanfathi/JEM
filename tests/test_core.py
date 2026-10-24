import math

import pytest
import torch

from jem.config import ModelConfig, make_config
from jem.energy import ConcatEnergy, OptionLinearEnergy
from jem.envs import make_world
from jem.inference import _grad, infer_newton, newton_solve, option_objective


def _features(b=4, n=5, k=33, d=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 2.0 * torch.randn(b, n, k, d, generator=g, dtype=torch.float64)


def test_newton_reaches_optimum():
    U = _features()
    z, residual = newton_solve(U, prec=0.5, iters=25)
    assert residual.max() < 1e-9
    # Any perturbation increases the objective (strict convexity).
    bump = 1e-3 * torch.randn(U.shape[0], 8, U.shape[-1], dtype=U.dtype)
    f0 = option_objective(z[:, None], U, 0.5)
    f1 = option_objective(z[:, None] + bump, U, 0.5)
    assert (f1 > f0).all()


def test_newton_is_init_free_and_matches_gradient_descent():
    U = _features(seed=1)
    z, _ = newton_solve(U, prec=1.0, iters=25)
    zg = torch.zeros_like(z)
    for _ in range(5000):
        zg = zg - 0.02 * _grad(zg, U, 1.0)[0]
    assert torch.allclose(z, zg, atol=1e-6)


def test_implicit_gradient_matches_finite_differences():
    U0, U1 = _features(seed=2), _features(seed=3)
    w = torch.randn(
        U0.shape[0], U0.shape[-1], dtype=U0.dtype, generator=torch.Generator().manual_seed(4)
    )
    theta = torch.tensor(0.7, dtype=U0.dtype, requires_grad=True)

    z, _ = infer_newton(theta * U0 + U1, prec=1.0, iters=30, implicit_grad=True)
    (grad,) = torch.autograd.grad((w * z).sum(), theta)

    eps = 1e-4  # smaller eps amplifies solver round-off
    zp, _ = newton_solve((theta.detach() + eps) * U0 + U1, 1.0, 30)
    zm, _ = newton_solve((theta.detach() - eps) * U0 + U1, 1.0, 30)
    fd = ((w * zp).sum() - (w * zm).sum()) / (2 * eps)
    assert torch.allclose(grad, fd, rtol=1e-5, atol=1e-7)
    assert torch.allclose(z, newton_solve((theta * U0 + U1).detach(), 1.0, 30)[0])


@pytest.mark.parametrize("norm", ["sphere", "tanh", "none"])
def test_composition_is_vector_addition(norm):
    torch.manual_seed(0)
    model = OptionLinearEnergy(
        4, 2, ModelConfig(option_dim=5, hidden=32, depth=2, feature_norm=norm)
    )
    obs, acts = torch.randn(7, 4), torch.rand(7, 11, 2) * 2 - 1
    z1, z2 = torch.randn(7, 5), torch.randn(7, 5)
    lhs = model.logits(obs, z1, acts) + model.logits(obs, z2, acts)
    rhs = model.logits(obs, z1 + z2, acts)
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_bounded_energy_scale():
    model = OptionLinearEnergy(4, 2, ModelConfig(option_dim=6, hidden=32, depth=2))
    obs, acts, z = torch.randn(9, 4), torch.rand(9, 50, 2) * 2 - 1, 3 * torch.randn(9, 6)
    bound = model.scale * z.norm(dim=-1, keepdim=True)
    assert (model.logits(obs, z, acts).abs() <= bound + 1e-4).all()


def test_concat_energy_broadcasts_candidates():
    model = ConcatEnergy(4, 2, ModelConfig(kind="mlp", option_dim=3, hidden=16, depth=2))
    obs, z, acts = torch.randn(5, 6, 4), torch.randn(5, 6, 3), torch.rand(5, 6, 9, 2)
    out = model.logits(obs, z, acts)
    assert out.shape == (5, 6, 9)
    assert torch.allclose(out[:, :, 3], model.logits(obs, z, acts[:, :, 3:4])[..., 0], atol=1e-6)


def test_expert_densities_normalise():
    world = make_world(make_config("jem").env)
    grid = torch.linspace(-1, 1, 801, dtype=torch.float64)
    cell = (grid[1] - grid[0]) ** 2
    acts = torch.stack(torch.meshgrid(grid, grid, indexing="ij"), -1).reshape(-1, 2)
    obs = torch.tensor([[0.3, -0.2, -0.1, 0.25]], dtype=torch.float64).expand(len(acts), 4)
    for e in range(world.num_experts):
        expert = torch.full((len(acts),), e)
        mass = world.expert_log_prob(obs, expert, acts).exp().sum() * cell
        assert abs(mass.item() - 1.0) < 0.02, (e, mass.item())


def test_expert_samples_follow_heading():
    world = make_world(make_config("jem").env)
    gen = torch.Generator().manual_seed(0)
    obs = world.reset(4000, gen)
    for e in range(world.num_experts):
        expert = torch.full((4000,), e)
        act = world.expert_act(obs, expert, gen)
        cos = torch.nn.functional.cosine_similarity(act, world.expert_mean_dir(obs, expert), dim=-1)
        assert abs(cos.mean().item() - math.exp(-0.5 * 0.5**2)) < 0.02


def test_config_overrides():
    cfg = make_config("legacy", ["model.option_dim=3", "eval.context_lens=1,2", "train.amp=true"])
    assert cfg.model.kind == "mlp" and cfg.model.option_dim == 3
    assert cfg.eval.context_lens == (1, 2) and cfg.train.amp is True
    with pytest.raises(KeyError):
        make_config("jem", ["model.nope=1"])


def test_weighted_newton_matches_duplicated_steps():
    U = _features(n=4, seed=5)
    w = torch.tensor([1.0, 2.0, 1.0, 1.0], dtype=U.dtype).expand(U.shape[0], 4)
    z_w, res = newton_solve(U, 1.0, 25, w)
    z_dup, _ = newton_solve(torch.cat([U, U[:, 1:2]], 1), 1.0, 25)
    assert res.max() < 1e-9 and torch.allclose(z_w, z_dup, atol=1e-10)
    ones = torch.ones(U.shape[0], 4, dtype=U.dtype)
    assert torch.allclose(newton_solve(U, 1.0, 25, ones)[0], newton_solve(U, 1.0, 25)[0])


def test_weighted_implicit_gradient_matches_finite_differences():
    U0, U1 = _features(seed=6), _features(seed=7)
    w = torch.rand(U0.shape[:2], generator=torch.Generator().manual_seed(8), dtype=U0.dtype)
    v = torch.randn(
        U0.shape[0], U0.shape[-1], dtype=U0.dtype, generator=torch.Generator().manual_seed(9)
    )
    theta = torch.tensor(0.4, dtype=U0.dtype, requires_grad=True)
    z, _ = infer_newton(theta * U0 + U1, 1.0, 30, True, w)
    (grad,) = torch.autograd.grad((v * z).sum(), theta)
    eps = 1e-3  # one problem stalls at a 1e-10 residual, so smaller eps is noisier
    zp, _ = newton_solve((0.4 + eps) * U0 + U1, 1.0, 30, w)
    zm, _ = newton_solve((0.4 - eps) * U0 + U1, 1.0, 30, w)
    fd = ((v * zp).sum() - (v * zm).sum()) / (2 * eps)
    assert torch.allclose(grad, fd, rtol=2e-4, atol=1e-7)


def test_fair_snippets_give_the_analytic_bayes_rate():
    # One step, four experts 90 degrees apart, Gaussian heading noise 0.5: Bayes accuracy is
    # P(|noise| < 45°) = 2 Phi(pi/2) - 1 = 0.8838 when the state carries no label information.
    from jem.data import collect
    from jem.evaluate import fair_snippets

    world = make_world(make_config("jem").env)
    gen = torch.Generator().manual_seed(0)
    demos = collect(world, 256, 32, gen)
    w = fair_snippets(world, demos, 40000, 1, gen)
    experts = torch.arange(4).expand(*w.expert.shape, 4)
    ll = world.expert_log_prob(w.obs[..., None, :], experts, w.act[..., None, :]).sum(1)
    acc = (ll.argmax(-1) == w.expert[:, 0]).float().mean().item()
    assert abs(acc - (2 * 0.5 * (1 + math.erf(math.pi / 2 / math.sqrt(2))) - 1)) < 0.01


def test_true_product_of_experts_reference():
    # The weighted product of Gaussian-heading experts 90° apart turns 90° w / (1 + w).
    from jem.evaluate import angle_deg, mean_action, true_logits
    from jem.inference import Proposals

    world = make_world(make_config("jem").env)
    gen = torch.Generator().manual_seed(1)
    states = world.reset(256, gen)
    props = Proposals(2, -1, 1, 16384, "cpu", 3).sample(gen)
    t = true_logits(world, states, props)
    i, j = world.expert_names.index("toward"), world.expert_names.index("cw")
    dir_i = world.expert_mean_dir(states, torch.full((256,), i))
    for w in (0.25, 0.5, 2.0, 4.0):
        turn = angle_deg(mean_action(t[:, i] + w * t[:, j], props), dir_i).mean().item()
        assert abs(turn - 90 * w / (1 + w)) < 1.0, (w, turn)


def test_support_edge_is_finite_in_fp32():
    world = make_world(make_config("jem").env)
    obs = torch.tensor([[0.3, -0.2, -0.1, 0.25]]).expand(360, 4)
    angles = torch.linspace(-math.pi, math.pi, 360)
    act = 0.5 * torch.stack([angles.cos(), angles.sin()], -1)  # speed exactly speed_min
    assert torch.isfinite(world.expert_log_prob(obs, torch.zeros(360, dtype=torch.long), act)).all()


def test_cw_orbits_clockwise():
    world = make_world(make_config("jem").env)
    obs = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # agent at (1, 0), target at the origin
    for name, sign in (("cw", -1), ("ccw", 1)):
        v = world.expert_mean_dir(obs, torch.tensor([world.expert_names.index(name)]))[0]
        assert math.copysign(1, v[1].item()) == sign  # angular momentum x*vy - y*vx
