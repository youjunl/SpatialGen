from typing import Any

from src.data.example_loader import ExampleDataset

__all__ = ["ExampleDataset", "MixDataset"]


def __getattr__(name: str) -> Any:
    # MixDataset pulls in the training-only loader (pytorch3d / open3d), so
    # import it lazily.  Inference only needs ExampleDataset.
    if name == "MixDataset":
        from src.data.mix_loader import MixDataset

        return MixDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
