"""Experiment orchestration: train + evaluate + record a verdict.

Spec §1 forbids logic in ``scripts/``, so each script in ``scripts/`` parses arguments
and calls one function from here. Keeping orchestration in the package also means the
gate experiments are importable from tests, which is how their wiring gets covered
without paying for a full training run.
"""
