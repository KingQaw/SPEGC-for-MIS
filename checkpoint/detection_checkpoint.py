# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
RECONSTRUCTED MODULE -- not part of the original SPEGC release.

`engine/trainer.py` line 49 performs, at module import time:

    from checkpoint.detection_checkpoint import DetectionTSCheckpointer

The repository ships no `checkpoint/` package (it was swallowed by the
unanchored `checkpoint/` pattern in .gitignore), so *every* entry point --
including plain `train_net.py --eval-only` -- failed with ModuleNotFoundError
before any model code ran.

detectron2's own ``DetectionCheckpointer`` already handles arbitrary
``nn.Module`` graphs, which includes SPEGC's ``EnsembleTSModel`` wrapper
(``modelTeacher`` / ``modelStudent`` children are ordinary submodules, so
``state_dict()`` / ``load_state_dict()`` round-trip correctly).  The original
OpenMatch-derived class of this name only added Caffe2-blob handling and a
``_load_file`` override, neither of which is required here.

This subclass exists so the public name imported by ``engine/trainer.py``
resolves, while inheriting the well-tested detectron2 behaviour.
"""

from detectron2.checkpoint import DetectionCheckpointer

__all__ = ["DetectionTSCheckpointer"]


class DetectionTSCheckpointer(DetectionCheckpointer):
    """Checkpointer used by ``SPEGCTrainer`` for the teacher/student ensemble.

    Behaves exactly like :class:`detectron2.checkpoint.DetectionCheckpointer`;
    it accepts a ``DetectionCheckpointer``-compatible ``nn.Module`` (a plain
    model, or the teacher/student ensemble wrapper) and preserves the usual
    ``resume_or_load`` / ``save`` / ``has_checkpoint`` semantics.
    """
