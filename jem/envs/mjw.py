"""Batched MuJoCo Warp physics as a pure function of torch tensors.

    qpos', qvel' = MjwSim(mjm, device, frame_skip).step(qpos, qvel, ctrl[, mocap_pos])

advances n independent worlds by `frame_skip` physics steps of `mjm.opt.timestep` with ctrl
held fixed. The simulator keeps no episode state, so one instance serves demonstration
collection, closed-loop evaluation and model-predictive control alike: cloning a world is
indexing a tensor.

Each call writes its inputs into the device Data through zero-copy views, replays one
captured CUDA graph on torch's current stream and returns copies of the result. Nothing
carries over from earlier calls:
  * time restarts at 0, and the first physics step starts the constraint solver cold (from
    the unconstrained acceleration, as with mjDSBL_WARMSTART); later steps of the same call
    warm-start as usual;
  * contacts and constraint rows are recomputed by every physics step, and qfrc_applied,
    xfrc_applied and mocap_quat keep their model defaults;
  * models with state outside (qpos, qvel) -- actuator activations, delay buffers, sleeping
    islands -- are rejected.

Each batch size n gets its own Data and graph, built on first use and kept (the first build
also compiles MuJoCo Warp's kernels for the model; Warp caches them on disk). Contact
buffers hold nconmax contacts per world, pooled over the batch, so one world may use more
if others use fewer; constraint buffers hold at most njmax rows per world (a pyramidal
condim-3 contact takes 4, an active joint limit 1). What does not fit is dropped and flagged
per world; `overflow()` reads the flags. Physics is float32 and not differentiable: step runs
under no_grad, and its outputs never require grad.

Worlds with several active contacts (or joint limits) are not bitwise reproducible: MuJoCo
Warp 3.17 allocates constraint rows with atomics and never sorts contacts
(DeterminismType.CONTACTS is a stub), so the solver's summation order varies between runs.
Repeated calls then differ in qpos by ~1e-7, rarely by ~1e-3. Worlds with at most one
contact and one active limit repeat exactly.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from torch import Tensor

# Hide Warp's init banner and its hundreds of per-kernel load messages; warnings still show.
wp.config.log_level = max(wp.config.log_level, wp.LOG_WARNING)

# The pooled contact buffer overflowing is printed by the GPU, once per physics step; every
# other flag (per-world rows, solver iteration limits) is only recorded for `overflow()`.
PRINTED_OVERFLOWS = mjw.OverflowType.BROADPHASE | mjw.OverflowType.NARROWPHASE


@dataclass
class _Batch:
    data: mjw.Data
    graph: wp.Graph
    qpos: Tensor  # zero-copy views of `data`
    qvel: Tensor
    ctrl: Tensor
    mocap_pos: Tensor
    overflow: Tensor


class MjwSim:
    def __init__(
        self,
        mjm: mujoco.MjModel,
        device: torch.device | str = "cuda",
        frame_skip: int = 1,
        *,
        nconmax: int = 32,
        njmax: int = 128,
    ):
        if mjm.na or mjm.nhistory:
            raise ValueError("actuator activations and delays are state outside (qpos, qvel)")
        if mjm.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP:
            raise ValueError("sleeping islands are state outside (qpos, qvel)")
        if min(frame_skip, nconmax, njmax) < 1:
            raise ValueError("frame_skip, nconmax and njmax must be positive")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("MuJoCo Warp needs a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.mjm, self.device, self.frame_skip = mjm, device, frame_skip
        self.nconmax, self.njmax = nconmax, njmax
        self.nq, self.nv, self.nu, self.nmocap = mjm.nq, mjm.nv, mjm.nu, mjm.nmocap
        self.dt = mjm.opt.timestep * frame_skip

        self._wp_device = wp.get_device(f"cuda:{device.index}")  # initialises Warp
        with wp.ScopedDevice(self._wp_device):
            self._model = mjw.put_model(mjm)
        self._model.opt.warn_overflow = int(PRINTED_OVERFLOWS)
        self._cold = copy.copy(self._model)
        self._cold.opt = copy.copy(self._model.opt)
        self._cold.opt.disableflags |= mjw.DisableBit.WARMSTART
        self._mocap_pos0 = torch.as_tensor(
            mujoco.MjData(mjm).mocap_pos, dtype=torch.float32, device=device
        )
        self._batches: dict[int, _Batch] = {}
        self._streams: dict[int, wp.Stream] = {}

    @torch.no_grad()
    def step(
        self, qpos: Tensor, qvel: Tensor, ctrl: Tensor, mocap_pos: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """(n, nq), (n, nv), (n, nu)[, (n, nmocap, 3)] -> next (qpos, qvel), new tensors.

        mocap_pos defaults to the model's mocap body positions. The first call for each n
        builds its Data and graph; later calls never synchronise the host (only a full CUDA
        launch queue, a few thousand kernels deep, makes them wait). Calls with the same n
        share one Data, so they must not overlap on different streams.
        """
        n = qpos.shape[0]
        if mocap_pos is None:
            mocap_pos = self._mocap_pos0.expand(n, -1, -1)
        self._check("qpos", qpos, (n, self.nq))
        self._check("qvel", qvel, (n, self.nv))
        self._check("ctrl", ctrl, (n, self.nu))
        self._check("mocap_pos", mocap_pos, (n, self.nmocap, 3))

        b = self._batch(n)
        b.qpos.copy_(qpos)
        b.qvel.copy_(qvel)
        b.ctrl.copy_(ctrl)
        if self.nmocap:
            b.mocap_pos.copy_(mocap_pos)
        wp.capture_launch(b.graph, stream=self._stream())
        return b.qpos.clone(), b.qvel.clone()

    def overflow(self, clear: bool = True) -> mjw.OverflowType:
        """OR of the flags raised by any world since the last clear. Synchronises the host.

        Buffer flags (NEFC: a world exceeded njmax; BROADPHASE/NARROWPHASE/CCD: the batch
        exceeded its contact pool) mean dropped constraints. ITERATIONS/LS_ITERATIONS only
        mean the solver stopped at its iteration cap.
        """
        torch.cuda.synchronize(self.device)  # steps may have run on any stream
        flags = 0
        for b in self._batches.values():
            flags |= int(np.bitwise_or.reduce(b.overflow.cpu().numpy()))
            if clear:
                b.overflow.zero_()
        torch.cuda.synchronize(self.device)
        return mjw.OverflowType(flags)

    def _check(self, name: str, x: Tensor, shape: tuple[int, ...]):
        if tuple(x.shape) != shape or x.device != self.device:
            raise ValueError(
                f"{name}: got {tuple(x.shape)} on {x.device}, need {shape} on {self.device}"
            )

    def _stream(self) -> wp.Stream:
        """Torch's current stream as a Warp stream: graphs run in order with torch's work."""
        stream = torch.cuda.current_stream(self.device)
        if stream.cuda_stream not in self._streams:
            self._streams[stream.cuda_stream] = wp.stream_from_torch(stream)
        return self._streams[stream.cuda_stream]

    def _batch(self, n: int) -> _Batch:
        if n in self._batches:
            return self._batches[n]
        if n < 1:
            raise ValueError("a batch needs at least one world")
        with wp.ScopedDevice(self._wp_device):
            d = mjw.make_data(self.mjm, nworld=n, nconmax=self.nconmax, njmax=self.njmax)
            mjw.step(self._cold, d)  # compile and load every kernel outside the capture
            mjw.step(self._model, d)
            with wp.ScopedCapture() as capture:
                d.time.zero_()
                mjw.step(self._cold, d)
                for _ in range(self.frame_skip - 1):
                    mjw.step(self._model, d)
            wp.capture_launch(capture.graph)  # instantiates the graph
            d.overflow.zero_()
            wp.synchronize_device()
        view = wp.to_torch
        self._batches[n] = b = _Batch(
            data=d,
            graph=capture.graph,
            qpos=view(d.qpos),
            qvel=view(d.qvel),
            ctrl=view(d.ctrl),
            mocap_pos=view(d.mocap_pos),
            overflow=view(d.overflow),
        )
        return b
