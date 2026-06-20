"""ElemeNet: multiscale molecular machine learning across the periodic table."""

# Backward-compatibility shim for the package rename ``elemnet`` -> ``elemenet``.
# Checkpoints saved by earlier versions pickle their model objects under module
# paths such as ``elemnet.model`` / ``elemnet.encoder``. Without a redirect,
# unpickling those ``.pt`` files (``checkpoint["model"]`` in inference and
# resume) fails with ``ModuleNotFoundError: No module named 'elemnet'``.
#
# Importing the model stack and aliasing each loaded submodule under the legacy
# name makes the unpickler resolve to the *same* module (and therefore the same
# class) objects, keeping ``isinstance`` checks valid for objects restored from
# old checkpoints. Only the modules a checkpoint references are pulled in here,
# so heavier optional dependencies (e.g. optuna, rdkit) are not imported eagerly.
import importlib
import sys

_LEGACY_NAME = "elemnet"

sys.modules.setdefault(_LEGACY_NAME, sys.modules[__name__])
# Pull in the encoder/readout classes that pickled model objects reference.
try:
    importlib.import_module(__name__ + ".model")
except Exception:
    pass
for _name in list(sys.modules):
    if _name == __name__ or _name.startswith(__name__ + "."):
        _legacy = _LEGACY_NAME + _name[len(__name__):]
        sys.modules.setdefault(_legacy, sys.modules[_name])
