# Test Fixtures

This directory stores small deterministic inputs for configuration, lifecycle, and frame-pipeline tests. Fixtures must be reviewable, reproducible, and free of proprietary game assets.

`quality-baselines.txt` is the stable metric catalog for deterministic motion and guidance fixtures. Absolute quality gates are authoritative: GPU estimator measurements must remain finite and within the shared thresholds used by the motion and temporal tests.
