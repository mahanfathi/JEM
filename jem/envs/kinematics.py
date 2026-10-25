"""Forward kinematics and Jacobian of a serial chain of hinge joints (batched, torch).

The chain runs from the world body to the body that carries a site of a compiled MjModel.
MuJoCo places each body by a fixed offset from its parent (body_pos, body_quat) and then by
its hinges. Hinge j has unit axis a_j and anchor c_j in the body frame, and angle
th_j = q_j - qpos0_j; with K_j = [a_j]_x (Rodrigues),

    H_j(th) = [[R, c_j - R c_j], [0, 1]],    R = I + sin(th) K_j + (1 - cos th) K_j^2.

Folding the fixed offsets A_j that precede hinge j into it, each factor is affine in
(sin th, 1 - cos th), so all J of them are formed at once from constant 4x4 matrices:

    G_j(th) = A_j H_j(th) = A_j + sin(th) A_j N_j + (1 - cos th) A_j N'_j,
    N_j = [[K_j, -K_j c_j], [0, 0]],    N'_j = [[K_j^2, -K_j^2 c_j], [0, 0]].

T_j = G_1 ... G_j is the world pose of the frame that hinge j moves, and the site is at
T_J B s (B: the fixed offsets after the last hinge, s: the site offset). A hinge leaves its
own anchor and axis in place, so

    o_j = T_j c_j,    w_j = rot(T_j) a_j,    d site / d q_j = w_j x (site - o_j).

This is mj_kinematics + mj_jacSite restricted to the chain (equal to round-off), batched
over any leading shape, in the dtype and on the device of q, without a simulator. Products
are broadcast sums rather than matmuls, so float32 stays exact when TF32 matmuls are on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    import mujoco

HINGE = 3  # mujoco.mjtJoint.mjJNT_HINGE


def rigid(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """4x4 transform of a MuJoCo offset (pos, quaternion w x y z)."""
    w, x, y, z = np.asarray(quat, dtype=np.float64) / np.linalg.norm(quat)
    out = np.eye(4)
    out[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    out[:3, 3] = pos
    return out


def hinge_terms(axis: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, N') with H(th) = I + sin(th) N + (1 - cos th) N'."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    n1, n2 = np.zeros((4, 4)), np.zeros((4, 4))
    n1[:3, :3], n1[:3, 3] = k, -k @ anchor
    n2[:3, :3], n2[:3, 3] = k @ k, -(k @ k) @ anchor
    return n1, n2


def compose(a: Tensor, b: Tensor) -> Tensor:
    """a @ b for (..., 4, 4) transforms."""
    return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)


def rotate(t: Tensor, v: Tensor) -> Tensor:
    """Rotate vectors v (..., 3) by the rotation part of t (..., 4, 4)."""
    return (t[..., :3, :3] * v.unsqueeze(-2)).sum(-1)


def transform(t: Tensor, p: Tensor) -> Tensor:
    """Map points p (..., 3) through t (..., 4, 4)."""
    return rotate(t, p) + t[..., :3, 3]


class ChainConstants(NamedTuple):
    g0: Tensor  # (J, 4, 4) A_j
    g1: Tensor  # (J, 4, 4) A_j N_j
    g2: Tensor  # (J, 4, 4) A_j N'_j
    qpos0: Tensor  # (J,)
    anchor: Tensor  # (J, 3) c_j
    axis: Tensor  # (J, 3) a_j
    tail: Tensor  # (4, 4) B s: site pose in the last hinge's frame
    root: Tensor  # (3,) origin of the chain's first body


class SerialChain:
    """Hinge chain from the world body to a site of `mjm`.

    q (..., J) holds the angles of every hinge between the world and the site, root to tip:
    q = qpos[..., chain.qpos_adr]. Jacobian column j belongs to dof chain.dof_adr[j].
    """

    def __init__(self, mjm: mujoco.MjModel, site: str, device: torch.device | str = "cpu"):
        sid = mjm.site(site).id
        bodies = [int(mjm.site_bodyid[sid])]
        while mjm.body_parentid[bodies[-1]] != 0:
            bodies.append(int(mjm.body_parentid[bodies[-1]]))
        bodies.reverse()

        fixed, g0, g1, g2, joints = np.eye(4), [], [], [], []
        for b in bodies:
            fixed = fixed @ rigid(mjm.body_pos[b], mjm.body_quat[b])
            for j in range(mjm.body_jntadr[b], mjm.body_jntadr[b] + mjm.body_jntnum[b]):
                if mjm.jnt_type[j] != HINGE:
                    raise ValueError(f"joint {mjm.joint(j).name!r} on the chain is not a hinge")
                n1, n2 = hinge_terms(mjm.jnt_axis[j], mjm.jnt_pos[j])
                g0.append(fixed)
                g1.append(fixed @ n1)
                g2.append(fixed @ n2)
                joints.append(j)
                fixed = np.eye(4)
        if not joints:
            raise ValueError(f"no joints between the world and site {site!r}")

        self.site_name = site
        self.joint_names = tuple(mjm.joint(j).name for j in joints)
        self.qpos_adr = tuple(int(mjm.jnt_qposadr[j]) for j in joints)
        self.dof_adr = tuple(int(mjm.jnt_dofadr[j]) for j in joints)
        self.njoint = len(joints)
        consts = ChainConstants(
            g0=np.stack(g0),
            g1=np.stack(g1),
            g2=np.stack(g2),
            qpos0=mjm.qpos0[list(self.qpos_adr)],
            anchor=mjm.jnt_pos[joints],
            axis=mjm.jnt_axis[joints],
            tail=fixed @ rigid(mjm.site_pos[sid], mjm.site_quat[sid]),
            root=mjm.body_pos[bodies[0]],
        )
        ref = ChainConstants(
            *(torch.as_tensor(np.array(c, dtype=np.float64), device=device) for c in consts)
        )
        self._consts = {(ref.g0.device, torch.float64): ref}

    def _const(self, q: Tensor) -> ChainConstants:
        """The chain's constants in q's dtype and on q's device (cached)."""
        key = (q.device, q.dtype)
        if key not in self._consts:
            ref = next(iter(self._consts.values()))
            self._consts[key] = ChainConstants(*(c.to(q.device, q.dtype) for c in ref))
        return self._consts[key]

    def transforms(self, q: Tensor) -> Tensor:
        """World poses T_j (..., J, 4, 4) of the frames the hinges move."""
        k = self._const(q)
        th = (q - k.qpos0)[..., None, None]
        g = k.g0 + th.sin() * k.g1 + 2 * (0.5 * th).sin().square() * k.g2  # 1 - cos, accurately
        out = [g[..., 0, :, :]]
        for j in range(1, self.njoint):
            out.append(compose(out[-1], g[..., j, :, :]))
        return torch.stack(out, dim=-3)

    def site(self, q: Tensor) -> Tensor:
        """Site position (..., 3)."""
        return transform(self.transforms(q)[..., -1, :, :], self._const(q).tail[:3, 3])

    def pose(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """Site position (..., 3) and orientation (..., 3, 3), as MuJoCo's site_xpos/xmat."""
        t = compose(self.transforms(q)[..., -1, :, :], self._const(q).tail)
        return t[..., :3, 3], t[..., :3, :3]

    def frames(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """World joint anchors o_j and unit axes w_j, each (..., J, 3)."""
        k, t = self._const(q), self.transforms(q)
        return transform(t, k.anchor), rotate(t, k.axis)

    def site_jacobian(self, q: Tensor) -> tuple[Tensor, Tensor]:
        """Site position (..., 3) and d site / d q (..., 3, J) from one pass."""
        k, t = self._const(q), self.transforms(q)
        x = transform(t[..., -1, :, :], k.tail[:3, 3])
        o, w = transform(t, k.anchor), rotate(t, k.axis)
        return x, torch.linalg.cross(w, x.unsqueeze(-2) - o, dim=-1).transpose(-1, -2)

    def jacobian(self, q: Tensor) -> Tensor:
        """d site / d q (..., 3, J): column j is w_j x (site - o_j)."""
        return self.site_jacobian(q)[1]

    def points(self, q: Tensor) -> Tensor:
        """Chain root, joint anchors and site (..., J + 2, 3), for stick-figure plots."""
        k, t = self._const(q), self.transforms(q)
        o = transform(t, k.anchor)
        x = transform(t[..., -1, :, :], k.tail[:3, 3])
        return torch.cat([k.root.expand(*o.shape[:-2], 1, 3), o, x.unsqueeze(-2)], dim=-2)
