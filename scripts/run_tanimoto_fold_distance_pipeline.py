from utils.tanimoto_fold_distance_utils import (
    TanimotoDistanceConfig,
    run_full_pipeline,
)

cfg = TanimotoDistanceConfig(
    task="hi",
    datasets_main=["drd2", "hiv", "sol"],
    models=["DT", "LR", "SVM"],
    fp_type="ecfp4",

    # Main analysis: complete pairwise Tanimoto distances.
    # Wasserstein is secondary and expensive, so we keep it off for this run.
    run_wasserstein=False,

    # Keep the main random-bit control as in the utility.
    n_random_bit_repeats=30,

    # Reasonable chunk size for Mac.
    pairwise_chunk_size=512,
)

run_full_pipeline(cfg)
