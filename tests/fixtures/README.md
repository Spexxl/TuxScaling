# Test Fixtures

This directory stores small deterministic inputs for configuration, lifecycle, and frame-pipeline tests. Fixtures must be reviewable, reproducible, and free of proprietary game assets.

`quality-baselines.txt` records the reference metrics and maximum regression deltas for the deterministic motion and guidance fixtures. Absolute quality gates are authoritative; GPU/driver-dependent measurements must also remain finite and within the shared thresholds used by the motion and temporal tests.
