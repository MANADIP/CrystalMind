"""
src/training — Training loop, losses, and LR scheduling.
"""

from src.training.trainer   import Trainer
from src.training.losses    import MultiTaskLoss

__all__ = [
    "Trainer",
    "MultiTaskLoss",
]
