from .presto import Presto, PrestoFineTuningModel

def construct_single_presto_input(*args, **kwargs):
    from .dataops.utils import construct_single_presto_input as _fn
    return _fn(*args, **kwargs)

__all__ = ["Presto", "PrestoFineTuningModel", "construct_single_presto_input"]
