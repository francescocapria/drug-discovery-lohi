import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.tanimoto_fold_distance_utils import (
    TanimotoDistanceConfig,
    run_full_pipeline,
)

cfg = TanimotoDistanceConfig(
    task="hi",
    datasets_main=["drd2", "hiv", "sol"],
    models=["DT", "LR", "SVM"],
    fp_type="ecfp4",
    expected_ecfp4_bits=1024,
    run_wasserstein=False,
    n_random_bit_repeats=30,
    pairwise_chunk_size=512,
)

run_full_pipeline(cfg)
