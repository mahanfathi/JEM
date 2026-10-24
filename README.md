# JEM: energy-based option discovery and composition

From unlabelled demonstrations of several behaviours, JEM learns a space of **options**
with a single energy function E(s, z, a). That one function does three jobs:

- **recognize** which option a short snippet of behaviour shows;
- **execute** an option as a stochastic policy;
- **compose** options into new behaviour:
  - concurrently, by adding energies;
  - over time, by planning in option space.

![options](docs/figures/options.png)

## The idea

The energy is **linear in the option**:

E(s, z, a) = −τ ⟨z, φ(s, a)⟩, with unit-norm learned features φ.

So p(a | s, z) is an exponential family over actions whose natural parameter is the
option z. Four things follow:

- **Composition is vector addition.** E(z₁) + E(z₂) = E(z₁ + z₂): the product of
  two option policies is again an option. Weights act as temperatures, −z negates an
  option, and z = 0 is the uniform policy.
- **Recognition is a convex problem.** The option behind a context of N steps is
  the MAP of a strongly convex objective. It is a log-partition function plus a
  Gaussian prior, so it has a single optimum. Damped Newton finds it in a few
  iterations on cached features. Training gets exact implicit-function-theorem
  gradients through it for the price of one extra Newton step.
- **The energy scale cannot drift.** |E| ≤ τ‖z‖.
- **No sampler in the training loop.** The model infers the option from 8 steps and
  is scored, with InfoNCE against fixed uniform proposals (as in Implicit BC), on
  the next 8 held-out steps. Only behaviour that persists over time helps with that
  prediction, which is what turns z into an option.

The method descends from Mordatch's *Concept Learning with Energy-Based Models* (2018)
and *Implicit Behavioral Cloning* (Florence et al., 2021). The repo is a 2026 PyTorch
rewrite of a 2022 JAX prototype (tag `legacy-jax-2022`). That prototype's algorithm is
available as the `legacy` preset.

## Testbed

A 2-D particle and a target. Four noisy scripted experts:

- move toward the target;
- move away from it;
- orbit clockwise;
- orbit counter-clockwise.

Each has heading noise of 0.5 rad, so one step is ambiguous, and its action density is
known in closed form. Every metric is therefore absolute: the true KL, the Bayes rate,
and the exact product of experts.

## Results

Default preset (`jem`), 10k steps, 3 seeds, about 6–7 minutes per run on a DGX Spark (GB10):

| | |
|---|---|
| Options rediscovered without labels (NMI of clustered z vs hidden expert) | 1.00 |
| Recognition accuracy from 1 / 2 / 4 / 8 steps (Bayes rate on the same snippets: 0.90 / 0.97 / 1.00 / 1.00) | 0.89 / 0.97 / 1.00 / 1.00 |
| Policy KL(expert ‖ option), nats | 0.05 |
| Zero-shot composition z_i + z_j vs the exact product of experts | 4.6° heading error |
| Weighted composition z_i + w z_j, w ∈ {¼, ½, 2, 4} | 4.8° heading error |
| Online tracking of hidden option switches (HMM over the library; Bayes 0.968) | 0.963 |
| Closed-loop agreement with the matched expert (experts' own value: 0.882) | 0.86 |

Early in training, the options organize into a 2-D compass in which opposite
behaviours are opposite vectors: cos(z_away, z_toward) ≈ −0.99 at 1k steps. Longer
training spreads them over more dimensions (−0.62 at 10k steps), trading that clean
geometry for a sharper fit.

![compositions](docs/figures/compositions.png)
![rollouts](docs/figures/rollouts.png)
![algebra](docs/figures/algebra.png)

**Composing over time.** The task is to reach the target around a concave obstacle that
no demonstration contained. Planning is MPC in the simulator, with 192 rollouts per
decision and a 40-step horizon, over 64 episodes × 3 seeds. Going straight for the
target gets trapped every time.

| Planner | Success |
|---|---|
| always "toward" | 0% |
| CEM over raw action sequences | 2% |
| CEM over the continuous option space | 47% |
| best of the discovered options and their pairwise sums | **88%** |

![downstream](docs/figures/downstream_cup.png)

**Open problem: demonstrations that switch options mid-trajectory.** The switching-aware
objective (`objective.hazard`) recovers recognition well. At hazard 1/32, NMI goes from
0.85 to 0.96 and tracking from 0.77 to 0.91. Policy fidelity still lags the
non-switching case.

## Quickstart

```bash
uv sync                                   # PyTorch cu130 wheels for aarch64 (see pyproject.toml)
uv run jem train --preset jem             # writes runs/<timestamp>-jem/
uv run jem dash                           # local dashboard: http://127.0.0.1:8765
```

More:

```bash
uv run jem presets                                    # jem, jem-em, jem-d2, jem-unnormed, legacy, legacy-tuned, smoke
uv run jem train --preset jem --set model.option_dim=2 env.switch_prob=0.03 objective.hazard=0.03
uv run jem eval runs/<run_id> --figures out/          # re-evaluate a checkpoint
uv run jem downstream runs/<run_id> --task cup --horizon 40
uv run jem table                                      # markdown table of all runs (seeds pooled)
./experiments/phase1.sh                               # the experiment suites (phase1-3)
uv run pytest
```

Use options programmatically:

```python
from jem.options import OptionLibrary

lib = OptionLibrary.load("runs/<run_id>")
spiral = lib["toward"] + lib["cw"]  # an inward spiral nobody demonstrated
traj = lib.rollout(spiral, lib.world.reset(8), steps=60)
belief = lib.recognize(obs, act, hazard=1 / 16)  # online option posterior
```

Each run directory holds:

- `config.json` and `status.json`;
- an append-only `metrics.jsonl`;
- figures;
- the last checkpoint, including the discovered option library.

`jem dash` reads these files live: charts with smoothing, log scale, and divergence
markers; figures by step; config diffs. It runs locally, with no external services.

## Layout

| Path | Contents |
|---|---|
| `jem/energy.py` | Option-linear energy (factored or joint features); the 2022 concat-MLP energy |
| `jem/inference.py` | Newton option inference with implicit gradients; Langevin inference (2022) |
| `jem/objectives.py` | The JEM objective (plain or switching-aware); the 2022 ML+KL+KLz objective |
| `jem/evaluate.py`, `jem/figures.py` | Metrics against true expert densities; figures |
| `jem/downstream.py` | Obstacle tasks and MPC planners |
| `jem/options.py` | `OptionLibrary`: recognize, act, compose, roll out |
| `jem/dashboard/` | The local dashboard (stdlib server and one HTML page) |
| `experiments/` | Reproducible experiment suites |
