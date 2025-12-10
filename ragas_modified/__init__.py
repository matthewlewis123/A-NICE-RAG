from ragas_modified.cache import CacheInterface, DiskCacheBackend, cacher
from ragas_modified.dataset_schema import EvaluationDataset, MultiTurnSample, SingleTurnSample
from ragas_modified.evaluation import evaluate
from ragas_modified.run_config import RunConfig

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown version"


__all__ = [
    "evaluate",
    "RunConfig",
    "__version__",
    "SingleTurnSample",
    "MultiTurnSample",
    "EvaluationDataset",
    "cacher",
    "CacheInterface",
    "DiskCacheBackend",
]


def __getattr__(name):
    if name == "experimental":
        try:
            import ragas_experimental as experimental  # type: ignore

            return experimental
        except ImportError:
            raise ImportError(
                "ragas.experimental requires installation: "
                "pip install ragas[experimental]"
            )
    raise AttributeError(f"module 'ragas' has no attribute '{name}'")
