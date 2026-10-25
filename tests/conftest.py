import os

# MuJoCo picks its OpenGL backend when first imported, by whichever test module comes
# first; recordings (tests/test_record.py) need the headless one.
os.environ.setdefault("MUJOCO_GL", "egl")
