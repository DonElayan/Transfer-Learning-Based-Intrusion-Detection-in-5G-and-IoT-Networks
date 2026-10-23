"""BoT-IoT (source) -> ToN-IoT (target) transfer-learning pipeline.

Stages (each one is a CLI module):
    python -m bot_ton_tl.prepare     raw CSVs -> harmonised, de-duplicated parquet
    python -m bot_ton_tl.diagnose    domain-shift report between the two datasets
    python -m bot_ton_tl.experiment  baselines + transfer-learning learning curves
"""

__version__ = "0.1.0"
