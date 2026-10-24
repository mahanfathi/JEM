"""Zero-dependency local dashboard for run directories written by `jem.runlog`."""

from .server import make_server, serve

__all__ = ["make_server", "serve"]
