# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
RECONSTRUCTED MODULE -- not part of the original SPEGC release.

`train_net.py` line 20 performs, at import time:

    import data.datasets.builtin

which exists purely for its registration side effects.  The repository ships no
`data/datasets/` package: the unanchored ``datasets/`` entry in .gitignore also
matched ``data/datasets/``, so the whole package was silently excluded from the
release commit (``git check-ignore -v data/datasets/builtin.py`` proves it).
The original contents are therefore unrecoverable and everything below is a
faithful reconstruction of the *contract* the rest of the code relies on.

What the rest of the code requires from this module
---------------------------------------------------
* ``detectron2.data.DatasetCatalog`` must know every name that appears in
  ``DATASETS.TRAIN`` / ``DATASETS.TEST`` / ``DATASETS.TRAIN_LABEL`` /
  ``DATASETS.TRAIN_UNLABEL``.  Otherwise ``get_detection_dataset_dicts``
  raises ``KeyError: Dataset 'X' is not registered!``.
* ``detectron2.data.MetadataCatalog`` must expose, for every ``DATASETS.TEST``
  name, both ``thing_classes`` and ``evaluator_type``.  ``build_evaluator``
  reads ``evaluator_type`` immediately, and only ``NotImplementedError`` is
  caught, so a missing key aborts evaluation.
* Registration is **lazy**: importing this module must not touch the disk, so
  that a missing dataset only fails if it is actually requested.

Layout assumed (see README.md and configs/)
-------------------------------------------
Fundus (documented in README.md)::

    datasets/Fundus/<name>_<split>.json
    datasets/Fundus/<name>/                # image root

Polyp datasets are referenced by configs/seg_res50fpn_source.yaml but are not
documented in README.md; the ``datasets/Polyp/`` layout below is INFERRED and
should be adjusted if the real release uses a different one.

Synthetic smoke-test data (created by tools/make_synthetic_fundus.py)::

    datasets/synthetic/{train,test}.json
    datasets/synthetic/images/
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

__all__ = ["register_all_spegc_datasets"]

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# NOTE: thing_classes is NOT hard-coded here.
# detectron2's load_coco_json derives thing_classes from the JSON's own
# "categories" and asserts that any pre-existing value matches:
#     AssertionError: Attribute 'thing_classes' in the metadata of 'X'
#     cannot be set to a different value!
# A guess here would therefore break every dataset whose category names differ
# (fundus = optic_disc/optic_cup with NUM_CLASSES=2, polyp = single class with
# NUM_CLASSES=1, synthetic = lesion), so the JSON stays the single source of
# truth.  evaluator_type -- the one key build_evaluator reads eagerly -- is set
# by register_coco_instances itself.

# --- real datasets -----------------------------------------------------------------
# Fundus target domains, per README.md.
_FUNDUS_DATASETS = ("Drishti_GS", "ORIGA", "REFUGE", "RIM_ONE_r3")
_FUNDUS_SPLITS = ("train", "test")
# REFUGE additionally ships a validation split used as a target domain in
# configs/test_segment.yaml ("REFUGE_Valid"), whose image root is REFUGE/.
_FUNDUS_EXTRA = (("REFUGE_Valid", "REFUGE_Valid.json", "REFUGE"),)

# Polyp datasets referenced by configs/seg_res50fpn_source.yaml (inferred layout).
_POLYP_DATASETS = ("BKAI", "CVC_ClinicDB", "ETIS_LaribPolypDB", "Kvasir_SEG")
_POLYP_SPLITS = ("train", "test")

# --- synthetic smoke-test datasets -------------------------------------------------
_SYNTH_DATASETS = ("train", "test")


def _release_root(*parts):
    return os.path.join(_REPO_ROOT, *parts)


def _register_coco(name, json_file, image_root):
    """Lazily register one COCO instance-segmentation dataset.

    ``register_coco_instances`` stores a thunk, so no file is opened here; the
    JSON is only parsed when the dataset is actually requested.  This keeps
    ``import data.datasets.builtin`` cheap and side-effect free even when the
    real datasets are absent.
    """
    if name in DatasetCatalog.list():
        return False
    register_coco_instances(
        name,
        {},
        json_file,
        image_root,
    )
    return True


def register_all_spegc_datasets():
    """Register every dataset name referenced by the shipped configs."""
    registered = []

    # Synthetic data for the CPU smoke test (tools/make_synthetic_fundus.py).
    for split in _SYNTH_DATASETS:
        name = "SynthFundus_{}".format(split)
        if _register_coco(
            name,
            _release_root("datasets", "synthetic", "{}.json".format(split)),
            _release_root("datasets", "synthetic", "images"),
        ):
            registered.append(name)

    # Real fundus target domains.
    for dataset in _FUNDUS_DATASETS:
        for split in _FUNDUS_SPLITS:
            name = "{}_{}".format(dataset, split)
            if _register_coco(
                name,
                _release_root("datasets", "Fundus", "{}.json".format(name)),
                _release_root("datasets", "Fundus", dataset),
            ):
                registered.append(name)

    for name, json_name, image_dir in _FUNDUS_EXTRA:
        if _register_coco(
            name,
            _release_root("datasets", "Fundus", json_name),
            _release_root("datasets", "Fundus", image_dir),
        ):
            registered.append(name)

    # Real polyp source-domain datasets.
    for dataset in _POLYP_DATASETS:
        for split in _POLYP_SPLITS:
            name = "{}_{}".format(dataset, split)
            if _register_coco(
                name,
                _release_root("datasets", "Polyp", "{}.json".format(name)),
                _release_root("datasets", "Polyp", dataset),
            ):
                registered.append(name)

    return registered


register_all_spegc_datasets()
