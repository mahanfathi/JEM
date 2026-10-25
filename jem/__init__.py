"""JEM: energy-based option discovery and composition."""

import os
import sys

# MuJoCo binds its OpenGL backend when it is first imported, and the arm's modules import
# it, so the headless default that `jem record` needs is set here, before any jem module
# can. Physics never needs it: MuJoCo skips its renderer if EGL fails to load. Linux only,
# since MuJoCo rejects MUJOCO_GL=egl at import elsewhere.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

__version__ = "0.2.0"
