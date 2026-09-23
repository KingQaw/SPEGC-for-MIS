# Compatibility shim for the SPEGC environment.
#
# Installed by tools/setup_env.sh into the environment's site-packages.
#
# torch 1.9.1 does, in torch/utils/tensorboard/__init__.py:
#
#     import tensorboard
#     from setuptools import distutils
#     LooseVersion = distutils.version.LooseVersion     # <-- attribute access
#
# `distutils.version` is a *submodule*; it is only reachable as an attribute of
# the `distutils` package if something imported it explicitly first. Nothing in
# this stack does (verified for the stdlib distutils, for setuptools' vendored
# setuptools._distutils, and after `import setuptools`), so detectron2's
# default_writers() -> TensorboardXWriter() -> `from torch.utils.tensorboard
# import SummaryWriter` raised:
#
#     AttributeError: module 'distutils' has no attribute 'version'
#
# Importing the submodule here, at interpreter start-up (site.py imports
# sitecustomize automatically), makes the attribute available for the rest of
# the process without patching any installed package.
import distutils.version  # noqa: F401
