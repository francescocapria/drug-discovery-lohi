"""
Utility library for Lo-Hi benchmark experiments.

Modules:
    fingerprints    - Molecular featurization (ECFP4, MACCS, RDKit topo, RDKit descriptors)
    metrics         - Hi (PR AUC) and Lo (Spearman) metrics
    cv_pipeline     - Nested cross-validation with inner model selection
    io_utils        - Load data, save results, manage paths
    config_loader   - YAML config parsing and validation
"""

