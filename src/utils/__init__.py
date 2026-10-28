"""Utility helpers.

Keep this package initializer import-safe.  Importing any ``src.utils.*``
submodule executes this file first; the original eager star-imports pulled in
optional heavy dependencies (ImageReward, kiui, wandb, open3d, pytorch3d) even
for inference, where none of them are needed.
"""
