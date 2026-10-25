import os
import sys

# MuJoCo picks its OpenGL backend when first imported, and some test modules import it
# before jem (whose __init__ sets the same default); recordings need the headless one.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")
