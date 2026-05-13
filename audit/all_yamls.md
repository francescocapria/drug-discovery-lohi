# ALL YAML CONFIGS
# Generated on: Wed May 13 12:58:17 AM CEST 2026
# Repository: /home/f.capria/drug-discovery-lohi


================================================================================
FILE: configs/hi/drd2/decision_tree/dt_ecfp4_drd2_hi.yaml
================================================================================

experiment:
  name: dt_ecfp4_drd2_hi
  task: hi
  dataset: drd2
fingerprint:
  type: ecfp4
model:
  name: dt
  fixed:
    random_state: 42
  search:
    max_depth:
    - 3
    - 5
    - 7
    - 10
    - 15
    - 20
    min_samples_split:
    - 2
    - 5
    - 10
    - 20
    min_samples_leaf:
    - 1
    - 2
    - 5
    - 10
    criterion:
    - gini
    - entropy
    class_weight:
    - null
    - balanced
    max_features:
    - sqrt
    - log2
    - null
    ccp_alpha:
    - 0.0
    - 0.0001
    - 0.001
    - 0.01
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/drd2/decision_tree/dt_maccs_drd2_hi.yaml
================================================================================

experiment:
  name: dt_maccs_drd2_hi
  task: hi
  dataset: drd2
fingerprint:
  type: maccs
model:
  name: dt
  fixed:
    random_state: 42
  search:
    max_depth:
    - 3
    - 5
    - 7
    - 10
    - 15
    - 20
    min_samples_split:
    - 2
    - 5
    - 10
    - 20
    min_samples_leaf:
    - 1
    - 2
    - 5
    - 10
    criterion:
    - gini
    - entropy
    class_weight:
    - null
    - balanced
    max_features:
    - sqrt
    - log2
    - null
    ccp_alpha:
    - 0.0
    - 0.0001
    - 0.001
    - 0.01
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/drd2/dummy/dummy_drd2_hi.yaml
================================================================================

experiment:
  name: dummy_drd2_hi
  task: hi
  dataset: drd2

fingerprint:
  type: ecfp4        

model:
  name: dummy
  fixed:
    strategy: most_frequent   
  search: {}                  

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/hi/drd2/gradient_boosting/gb_drd2_hi.yaml
================================================================================

experiment:
  name: gb_drd2_hi
  task: hi
  dataset: drd2

fingerprint:
  types: [maccs]

model:
  name: gb
  fixed:
    random_state: 42
  search:
    n_estimators: [100, 200, 300, 500]
    learning_rate: [0.01, 0.05, 0.1, 0.2]
    max_depth: [3, 4, 5, 6]
    min_samples_split: [2, 5, 10]
    min_samples_leaf: [1, 2, 4]
    subsample: [0.7, 0.8, 1.0]

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: random
  n_iter: 80
  random_state: 42

================================================================================
FILE: configs/hi/drd2/knn/knn_drd2_hi.yaml
================================================================================

  experiment:
    name: knn_ecfp4_drd2_hi
    task: hi
    dataset: drd2

  fingerprint:
    types: [ecfp4, maccs]

  model:
    name: knn
    fixed:
      algorithm: auto
    search:
      n_neighbors: [3, 5, 7, 11, 15, 21, 31, 51, 75, 101]
      metric: [jaccard]
      weights: [uniform, distance]

  cv:
    inner_k: 2
    scoring: average_precision
    search_strategy: grid
    random_state: 42

================================================================================
FILE: configs/hi/drd2/lr/lr_drd2_hi_rdkit_topo.yaml
================================================================================

experiment:
  name: lr_maccs_drd2_hi
  task: hi
  dataset: drd2
fingerprint:
  type: rdkit_topo
model:
  name: lr
  fixed:
    max_iter: 15000
    random_state: 42
    solver: saga
    penalty: elasticnet
  search:
    C:
    - 0.005
    - 0.01
    - 0.05
    - 0.1
    - 0.5
    - 1.0
    - 5.0
    - 10.0
    l1_ratio:
    - 0.0
    - 0.25
    - 0.5
    - 0.75
    - 1.0
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/drd2/lr/lr_drd2_hi.yaml
================================================================================

experiment:
  name: lr_maccs_drd2_hi
  task: hi
  dataset: drd2
fingerprint:
  types:
  - ecfp4
  - maccs
  - rdkit_desc
model:
  name: lr
  fixed:
    max_iter: 15000
    random_state: 42
    solver: saga
    penalty: elasticnet
  search:
    C:
    - 0.005
    - 0.01
    - 0.05
    - 0.1
    - 0.5
    - 1.0
    - 5.0
    - 10.0
    l1_ratio:
    - 0.0
    - 0.25
    - 0.5
    - 0.75
    - 1.0
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/drd2/random_forest/rf_drd2_hi.yaml
================================================================================

experiment:
  name: rf_drd2_hi
  task: hi
  dataset: drd2

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: rf
  fixed:
    random_state: 42
  search:
    n_estimators: [50, 100, 200, 300, 500, 800]
    max_depth: [null, 5, 10, 15, 20, 30, 50]
    min_samples_split: [2, 5, 10, 20]
    min_samples_leaf: [1, 2, 4, 8]
    max_features: [sqrt, log2, 0.1, 0.2, 0.3, 0.5]
    class_weight: [null, balanced]

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: random
  n_iter: 100
  random_state: 42

================================================================================
FILE: configs/hi/drd2/svm/svm_linear_drd2_hi.yaml
================================================================================

experiment:
  name: svm_linear_drd2_hi
  task: hi
  dataset: drd2
fingerprint:
  types:
  - ecfp4
  - maccs
model:
  name: svm
  fixed:
    kernel: linear
    probability: false
  search:
    C:
    - 0.001
    - 0.005
    - 0.01
    - 0.05
    - 0.1
    - 0.25
    - 0.5
    - 1.0
    - 2.0
    - 5.0
    - 10.0
    - 25.0
    - 50.0
    - 100.0
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/drd2/svm/svm_poly_drd2_hi.yaml
================================================================================

experiment:
  name: svm_poly_drd2_hi
  task: hi
  dataset: drd2
fingerprint:
  types:
  - ecfp4
  - maccs
  - rdkit_desc
model:
  name: svm
  fixed:
    kernel: poly
    probability: false
  search:
    C:
    - 0.1
    - 1.0
    - 5.0
    - 10.0
    degree:
    - 2
    - 3
    coef0:
    - 0.0
    - 1.0
    gamma:
    - scale
    - auto
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/drd2/svm/svm_rbf_drd2_hi.yaml
================================================================================

experiment:
  name: svm_rbf_drd2_hi
  task: hi
  dataset: drd2
fingerprint:
  types:
  - ecfp4
  - maccs
  - rdkit_desc
model:
  name: svm
  fixed:
    kernel: rbf
    probability: false
  search:
    C:
    - 0.1
    - 0.5
    - 1.0
    - 2.0
    - 5.0
    - 10.0
    gamma:
    - scale
    - auto
    - 0.001
    - 0.01
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/drd2/svm/svm_tanimoto_drd2_hi.yaml
================================================================================

experiment:
  name: svm_tanimoto_drd2_hi
  task: hi
  dataset: drd2
fingerprint:
  types:
  - ecfp4
  - maccs
model:
  name: svm
  fixed:
    kernel: tanimoto
    probability: false
  search:
    C:
    - 0.1
    - 0.5
    - 1.0
    - 2.0
    - 5.0
    - 10.0
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/drd2/svm/svm_tanimoto_ecfp4_drd2_hi_finetune.yaml
================================================================================

experiment:
  name: svm_tanimoto_ecfp4_drd2_hi_ft
  task: hi
  dataset: drd2
fingerprint:
  types:
  - ecfp4
model:
  name: svm
  fixed:
    kernel: tanimoto
    probability: false
  search:
    C:
    - 0.5
    - 0.75
    - 1.0
    - 1.5
    - 2.0
    - 3.0
    - 4.0
    - 5.0
    - 7.0
    - 10.0
    - 15.0
    - 20.0
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/drd2/xgboost/xgb_drd2_hi.yaml
================================================================================

experiment:
  name: xgb_drd2_hi
  task: hi
  dataset: drd2

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: xgb
  fixed:
    random_state: 42
    n_jobs: -1
  search:
    n_estimators: [100, 200, 300]
    learning_rate: [0.05, 0.1, 0.2]
    max_depth: [3, 4, 6]
    subsample: [0.8, 1.0]
    colsample_bytree: [0.8, 1.0]

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: random
  n_iter: 60
  random_state: 42

================================================================================
FILE: configs/hi/hiv/decision_tree/dt_ecfp4_hiv_hi.yaml
================================================================================

experiment:
  name: dt_ecfp4_hiv_hi
  task: hi
  dataset: hiv
fingerprint:
  type: ecfp4
model:
  name: dt
  fixed:
    random_state: 42
  search:
    max_depth:
    - 3
    - 5
    - 8
    - 10
    - 15
    min_samples_split:
    - 10
    - 20
    - 50
    min_samples_leaf:
    - 5
    - 10
    - 20
    - 50
    criterion:
    - gini
    - entropy
    class_weight:
    - null
    - balanced
    max_features:
    - sqrt
    - log2
    ccp_alpha:
    - 0.0
    - 0.0001
    - 0.001
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/hiv/decision_tree/dt_maccs_hiv_hi.yaml
================================================================================

experiment:
  name: dt_maccs_hiv_hi
  task: hi
  dataset: hiv
fingerprint:
  type: maccs
model:
  name: dt
  fixed:
    random_state: 42
  search:
    max_depth:
    - 3
    - 5
    - 8
    - 10
    - 15
    min_samples_split:
    - 10
    - 20
    - 50
    min_samples_leaf:
    - 5
    - 10
    - 20
    - 50
    criterion:
    - gini
    - entropy
    class_weight:
    - null
    - balanced
    max_features:
    - sqrt
    - log2
    ccp_alpha:
    - 0.0
    - 0.0001
    - 0.001
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/hiv/dummy/dummy_hiv_hi.yaml
================================================================================

experiment:
  name: dummy_hiv_hi
  task: hi
  dataset: hiv

fingerprint:
  type: ecfp4        

model:
  name: dummy
  fixed:
    strategy: most_frequent   
  search: {}                  

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/hi/hiv/gradient_boosting/gb_hiv_hi.yaml
================================================================================

experiment:
  name: gb_hiv_hi
  task: hi
  dataset: hiv

fingerprint:
  types: [maccs]

model:
  name: gb
  fixed:
    random_state: 42
  search:
    n_estimators: [100, 200, 300, 500]
    learning_rate: [0.01, 0.05, 0.1, 0.2]
    max_depth: [3, 4, 5, 6]
    min_samples_split: [2, 5, 10]
    min_samples_leaf: [1, 2, 4]
    subsample: [0.7, 0.8, 1.0]

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: random
  n_iter: 50
  random_state: 42

================================================================================
FILE: configs/hi/hiv/knn/knn_hiv_hi.yaml
================================================================================

experiment:
  name: knn_ecfp4_hiv_hi
  task: hi
  dataset: hiv

fingerprint:
  types: [ecfp4, maccs]

model:
  name: knn
  fixed:
    algorithm: auto
  search:
    n_neighbors: [3, 5, 7, 11, 15, 21, 31, 51, 75, 101]
    metric: [jaccard]
    weights: [uniform, distance]

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/hi/hiv/lr/lr_hiv_hi.yaml
================================================================================

experiment:
  name: lr_maccs_hiv_hi
  task: hi
  dataset: hiv
fingerprint:
  types:
  - ecfp4
  - maccs
  - rdkit_desc
model:
  name: lr
  fixed:
    max_iter: 40000
    random_state: 42
    solver: saga
    penalty: elasticnet
  search:
    C:
    - 0.005
    - 0.01
    - 0.05
    - 0.1
    - 0.5
    - 1.0
    - 5.0
    - 10.0
    l1_ratio:
    - 0.0
    - 0.25
    - 0.5
    - 0.75
    - 1.0
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/hiv/random_forest/rf_hiv_hi.yaml
================================================================================

experiment:
  name: rf_hiv_hi
  task: hi
  dataset: hiv

fingerprint:
  types: [rdkit_desc]

model:
  name: rf
  fixed:
    random_state: 42
  search:
    n_estimators: [50, 100, 200, 300, 500, 800]
    max_depth: [null, 5, 10, 15, 20, 30, 50]
    min_samples_split: [2, 5, 10, 20]
    min_samples_leaf: [1, 2, 4, 8]
    max_features: [sqrt, log2, 0.1, 0.2, 0.3, 0.5]
    class_weight: [null, balanced]

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: random
  n_iter: 100
  random_state: 42

================================================================================
FILE: configs/hi/hiv/svm/svm_linear_hiv_hi.yaml
================================================================================

experiment:
  name: svm_linear_hiv_hi
  task: hi
  dataset: hiv
fingerprint:
  types:
  - ecfp4
  - maccs
model:
  name: svm
  fixed:
    kernel: linear
    probability: false
  search:
    C:
    - 0.001
    - 0.005
    - 0.01
    - 0.05
    - 0.1
    - 0.25
    - 0.5
    - 1.0
    - 2.0
    - 5.0
    - 10.0
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/hiv/svm/svm_poly_hiv_hi.yaml
================================================================================

experiment:
  name: svm_poly_hiv_hi
  task: hi
  dataset: hiv
fingerprint:
  types:
  - ecfp4
  - maccs
  - rdkit_desc
model:
  name: svm
  fixed:
    kernel: poly
    probability: false
  search:
    C:
    - 0.1
    - 1.0
    - 5.0
    - 10.0
    degree:
    - 2
    - 3
    coef0:
    - 0.0
    - 1.0
    gamma:
    - scale
    - auto
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/hiv/svm/svm_rbf_hiv_hi.yaml
================================================================================

experiment:
  name: svm_rbf_hiv_hi
  task: hi
  dataset: hiv
fingerprint:
  types:
  - ecfp4
  - maccs
  - rdkit_desc
model:
  name: svm
  fixed:
    kernel: rbf
    probability: false
  search:
    C:
    - 0.1
    - 0.5
    - 1.0
    - 2.0
    - 5.0
    - 10.0
    gamma:
    - scale
    - auto
    - 0.001
    - 0.01
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/hiv/svm/svm_tanimoto_hiv_hi.yaml
================================================================================

experiment:
  name: svm_tanimoto_hiv_hi
  task: hi
  dataset: hiv
fingerprint:
  types:
  - ecfp4
  - maccs
model:
  name: svm
  fixed:
    kernel: tanimoto
    probability: false
  search:
    C:
    - 0.1
    - 0.5
    - 1.0
    - 2.0
    - 5.0
    - 10.0
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/hiv/xgboost/xgb_hiv_hi.yaml
================================================================================

experiment:
  name: xgb_hiv_hi
  task: hi
  dataset: hiv

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: xgb
  fixed:
    random_state: 42
    n_jobs: -1
  search:
    n_estimators: [100, 200, 300]
    learning_rate: [0.05, 0.1, 0.2]
    max_depth: [3, 4, 6]
    subsample: [0.8, 1.0]
    colsample_bytree: [0.8, 1.0]

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: random
  n_iter: 40
  random_state: 42

================================================================================
FILE: configs/hi/kdr/decision_tree/dt_ecfp4_kdr_hi.yaml
================================================================================

experiment:
  name: dt_ecfp4_kdr_hi
  task: hi
  dataset: kdr
fingerprint:
  type: ecfp4
model:
  name: dt
  fixed:
    random_state: 42
  search:
    max_depth:
    - 3
    - 5
    - 7
    - 10
    - 15
    - 20
    min_samples_split:
    - 2
    - 5
    - 10
    - 20
    min_samples_leaf:
    - 1
    - 2
    - 5
    - 10
    criterion:
    - gini
    - entropy
    class_weight:
    - null
    - balanced
    max_features:
    - sqrt
    - log2
    - null
    ccp_alpha:
    - 0.0
    - 0.0001
    - 0.001
    - 0.01
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/kdr/decision_tree/dt_maccs_kdr_hi.yaml
================================================================================

experiment:
  name: dt_maccs_kdr_hi
  task: hi
  dataset: kdr
fingerprint:
  type: maccs
model:
  name: dt
  fixed:
    random_state: 42
  search:
    max_depth:
    - 3
    - 5
    - 7
    - 10
    - 15
    - 20
    min_samples_split:
    - 2
    - 5
    - 10
    - 20
    min_samples_leaf:
    - 1
    - 2
    - 5
    - 10
    criterion:
    - gini
    - entropy
    class_weight:
    - null
    - balanced
    max_features:
    - sqrt
    - log2
    - null
    ccp_alpha:
    - 0.0
    - 0.0001
    - 0.001
    - 0.01
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/kdr/dummy/dummy_kdr_hi.yaml
================================================================================

experiment:
  name: dummy_kdr_hi
  task: hi
  dataset: kdr

fingerprint:
  type: ecfp4        

model:
  name: dummy
  fixed:
    strategy: most_frequent   
  search: {}                  

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/hi/kdr/gradient_boosting/gb_kdr_hi.yaml
================================================================================

experiment:
  name: gb_kdr_hi
  task: hi
  dataset: kdr

fingerprint:
  types: [maccs]

model:
  name: gb
  fixed:
    random_state: 42
  search:
    n_estimators: [100, 200, 300, 500]
    learning_rate: [0.01, 0.05, 0.1, 0.2]
    max_depth: [3, 4, 5, 6]
    min_samples_split: [2, 5, 10]
    min_samples_leaf: [1, 2, 4]
    subsample: [0.7, 0.8, 1.0]

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: random
  n_iter: 80
  random_state: 42

================================================================================
FILE: configs/hi/kdr/knn/knn_kdr_hi.yaml
================================================================================

experiment:
  name: knn_ecfp4_kdr_hi
  task: hi
  dataset: kdr

fingerprint:
  types: [ecfp4, maccs]

model:
  name: knn
  fixed:
    algorithm: auto
  search:
    n_neighbors: [3, 5, 7, 11, 15, 21, 31, 51, 75, 101]
    metric: [jaccard]
    weights: [uniform, distance]

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/hi/kdr/lr/lr_kdr_hi.yaml
================================================================================

experiment:
  name: lr_maccs_kdr_hi
  task: hi
  dataset: kdr
fingerprint:
  types:
  - ecfp4
  - maccs
  - rdkit_desc
model:
  name: lr
  fixed:
    max_iter: 15000
    random_state: 42
    solver: saga
    penalty: elasticnet
  search:
    C:
    - 0.005
    - 0.01
    - 0.05
    - 0.1
    - 0.5
    - 1.0
    - 5.0
    - 10.0
    l1_ratio:
    - 0.0
    - 0.25
    - 0.5
    - 0.75
    - 1.0
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/kdr/random_forest/rf_kdr_hi.yaml
================================================================================

experiment:
  name: rf_kdr_hi
  task: hi
  dataset: kdr

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: rf
  fixed:
    random_state: 42
  search:
    n_estimators: [50, 100, 200, 300, 500, 800]
    max_depth: [null, 5, 10, 15, 20, 30, 50]
    min_samples_split: [2, 5, 10, 20]
    min_samples_leaf: [1, 2, 4, 8]
    max_features: [sqrt, log2, 0.1, 0.2, 0.3, 0.5]
    class_weight: [null, balanced]

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: random
  n_iter: 100
  random_state: 42

================================================================================
FILE: configs/hi/kdr/svm/svm_linear_kdr_hi.yaml
================================================================================

experiment:
  name: svm_linear_kdr_hi
  task: hi
  dataset: kdr
fingerprint:
  types:
  - ecfp4
  - maccs
model:
  name: svm
  fixed:
    kernel: linear
    probability: false
  search:
    C:
    - 0.001
    - 0.005
    - 0.01
    - 0.05
    - 0.1
    - 0.25
    - 0.5
    - 1.0
    - 2.0
    - 5.0
    - 10.0
    - 25.0
    - 50.0
    - 100.0
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/kdr/svm/svm_poly_kdr_hi.yaml
================================================================================

experiment:
  name: svm_poly_kdr_hi
  task: hi
  dataset: kdr
fingerprint:
  types:
  - ecfp4
  - maccs
  - rdkit_desc
model:
  name: svm
  fixed:
    kernel: poly
    probability: false
  search:
    C:
    - 0.1
    - 1.0
    - 5.0
    - 10.0
    degree:
    - 2
    - 3
    coef0:
    - 0.0
    - 1.0
    gamma:
    - scale
    - auto
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/kdr/svm/svm_rbf_kdr_hi.yaml
================================================================================

experiment:
  name: svm_rbf_kdr_hi
  task: hi
  dataset: kdr
fingerprint:
  types:
  - ecfp4
  - maccs
  - rdkit_desc
model:
  name: svm
  fixed:
    kernel: rbf
    probability: false
  search:
    C:
    - 0.1
    - 0.5
    - 1.0
    - 2.0
    - 5.0
    - 10.0
    gamma:
    - scale
    - auto
    - 0.001
    - 0.01
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/kdr/svm/svm_tanimoto_kdr_hi.yaml
================================================================================

experiment:
  name: svm_tanimoto_kdr_hi
  task: hi
  dataset: kdr
fingerprint:
  types:
  - ecfp4
  - maccs
model:
  name: svm
  fixed:
    kernel: tanimoto
    probability: false
  search:
    C:
    - 0.1
    - 0.5
    - 1.0
    - 2.0
    - 5.0
    - 10.0
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/kdr/xgboost/xgb_kdr_hi.yaml
================================================================================

experiment:
  name: xgb_kdr_hi
  task: hi
  dataset: kdr

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: xgb
  fixed:
    random_state: 42
    n_jobs: -1
  search:
    n_estimators: [100, 200, 300]
    learning_rate: [0.05, 0.1, 0.2]
    max_depth: [3, 4, 6]
    subsample: [0.8, 1.0]
    colsample_bytree: [0.8, 1.0]

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: random
  n_iter: 60
  random_state: 42

================================================================================
FILE: configs/hi/sol/decision_tree/dt_ecfp4_sol_hi.yaml
================================================================================

experiment:
  name: dt_ecfp4_sol_hi
  task: hi
  dataset: sol
fingerprint:
  type: ecfp4
model:
  name: dt
  fixed:
    random_state: 42
  search:
    max_depth:
    - 3
    - 5
    - 7
    - 10
    - 15
    - 20
    min_samples_split:
    - 2
    - 5
    - 10
    - 20
    min_samples_leaf:
    - 1
    - 2
    - 5
    - 10
    criterion:
    - gini
    - entropy
    class_weight:
    - null
    - balanced
    max_features:
    - sqrt
    - log2
    - null
    ccp_alpha:
    - 0.0
    - 0.0001
    - 0.001
    - 0.01
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/sol/decision_tree/dt_maccs_sol_hi.yaml
================================================================================

experiment:
  name: dt_maccs_sol_hi
  task: hi
  dataset: sol
fingerprint:
  type: maccs
model:
  name: dt
  fixed:
    random_state: 42
  search:
    max_depth:
    - 3
    - 5
    - 7
    - 10
    - 15
    - 20
    min_samples_split:
    - 2
    - 5
    - 10
    - 20
    min_samples_leaf:
    - 1
    - 2
    - 5
    - 10
    criterion:
    - gini
    - entropy
    class_weight:
    - null
    - balanced
    max_features:
    - sqrt
    - log2
    - null
    ccp_alpha:
    - 0.0
    - 0.0001
    - 0.001
    - 0.01
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/sol/dummy/dummy_sol_hi.yaml
================================================================================

experiment:
  name: dummy_sol_hi
  task: hi
  dataset: sol

fingerprint:
  type: ecfp4       

model:
  name: dummy
  fixed:
    strategy: most_frequent  
  search: {}                  

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/hi/sol/gradient_boosting/gb_sol_hi.yaml
================================================================================

experiment:
  name: gb_sol_hi
  task: hi
  dataset: sol

fingerprint:
  types: [maccs]

model:
  name: gb
  fixed:
    random_state: 42
  search:
    n_estimators: [100, 200, 300, 500]
    learning_rate: [0.01, 0.05, 0.1, 0.2]
    max_depth: [3, 4, 5, 6]
    min_samples_split: [2, 5, 10]
    min_samples_leaf: [1, 2, 4]
    subsample: [0.7, 0.8, 1.0]

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: random
  n_iter: 80
  random_state: 42

================================================================================
FILE: configs/hi/sol/knn/knn_sol_hi.yaml
================================================================================

experiment:
  name: knn_ecfp4_sol__hi
  task: hi
  dataset: sol

fingerprint:
  types: [ecfp4, maccs]

model:
  name: knn
  fixed:
    algorithm: auto
  search:
    n_neighbors: [3, 5, 7, 11, 15, 21, 31, 51, 75, 101]
    metric: [jaccard]
    weights: [uniform, distance]

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/hi/sol/lr/lr_sol_hi.yaml
================================================================================

experiment:
  name: lr_maccs_sol_hi
  task: hi
  dataset: sol
fingerprint:
  types:
  - ecfp4
  - maccs
  - rdkit_desc
model:
  name: lr
  fixed:
    max_iter: 15000
    random_state: 42
    solver: saga
    penalty: elasticnet
  search:
    C:
    - 0.005
    - 0.01
    - 0.05
    - 0.1
    - 0.5
    - 1.0
    - 5.0
    - 10.0
    l1_ratio:
    - 0.0
    - 0.25
    - 0.5
    - 0.75
    - 1.0
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/sol/random_forest/rf_sol_hi.yaml
================================================================================

experiment:
  name: rf_sol_hi
  task: hi
  dataset: sol

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: rf
  fixed:
    random_state: 42
  search:
    n_estimators: [50, 100, 200, 300, 500, 800]
    max_depth: [null, 5, 10, 15, 20, 30, 50]
    min_samples_split: [2, 5, 10, 20]
    min_samples_leaf: [1, 2, 4, 8]
    max_features: [sqrt, log2, 0.1, 0.2, 0.3, 0.5]
    class_weight: [null, balanced]

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: random
  n_iter: 100
  random_state: 42

================================================================================
FILE: configs/hi/sol/svm/svm_linear_sol_hi.yaml
================================================================================

experiment:
  name: svm_linear_sol_hi
  task: hi
  dataset: sol
fingerprint:
  types:
  - ecfp4
  - maccs
model:
  name: svm
  fixed:
    kernel: linear
    probability: false
  search:
    C:
    - 0.001
    - 0.005
    - 0.01
    - 0.05
    - 0.1
    - 0.25
    - 0.5
    - 1.0
    - 2.0
    - 5.0
    - 10.0
    - 25.0
    - 50.0
    - 100.0
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/sol/svm/svm_poly_sol_hi.yaml
================================================================================

experiment:
  name: svm_poly_sol_hi
  task: hi
  dataset: sol
fingerprint:
  types:
  - ecfp4
  - maccs
  - rdkit_desc
model:
  name: svm
  fixed:
    kernel: poly
    probability: false
  search:
    C:
    - 0.1
    - 1.0
    - 5.0
    - 10.0
    degree:
    - 2
    - 3
    coef0:
    - 0.0
    - 1.0
    gamma:
    - scale
    - auto
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/sol/svm/svm_rbf_sol_hi.yaml
================================================================================

experiment:
  name: svm_rbf_sol_hi
  task: hi
  dataset: sol
fingerprint:
  types:
  - ecfp4
  - maccs
  - rdkit_desc
model:
  name: svm
  fixed:
    kernel: rbf
    probability: false
  search:
    C:
    - 0.1
    - 0.5
    - 1.0
    - 2.0
    - 5.0
    - 10.0
    gamma:
    - scale
    - auto
    - 0.001
    - 0.01
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/sol/svm/svm_tanimoto_sol_hi.yaml
================================================================================

experiment:
  name: svm_tanimoto_sol_hi
  task: hi
  dataset: sol
fingerprint:
  types:
  - ecfp4
  - maccs
model:
  name: svm
  fixed:
    kernel: tanimoto
    probability: false
  search:
    C:
    - 0.1
    - 0.5
    - 1.0
    - 2.0
    - 5.0
    - 10.0
    class_weight:
    - null
    - balanced
cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/hi/sol/xgboost/xgb_sol_hi.yaml
================================================================================

experiment:
  name: xgb_sol_hi
  task: hi
  dataset: sol

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: xgb
  fixed:
    random_state: 42
    n_jobs: -1
  search:
    n_estimators: [100, 200, 300]
    learning_rate: [0.05, 0.1, 0.2]
    max_depth: [3, 4, 6]
    subsample: [0.8, 1.0]
    colsample_bytree: [0.8, 1.0]

cv:
  inner_k: 2
  scoring: average_precision
  search_strategy: random
  n_iter: 60
  random_state: 42

================================================================================
FILE: configs/lo/drd2/decision_tree/dt_ecfp4_drd2_lo.yaml
================================================================================

experiment:
  name: dt_ecfp4_drd2_lo
  task: lo
  dataset: drd2
fingerprint:
  type: ecfp4
model:
  name: dt
  fixed:
    random_state: 42
  search:
    max_depth:
    - 3
    - 5
    - 7
    - 10
    - 15
    - 20
    - null
    min_samples_split:
    - 2
    - 5
    - 10
    - 20
    min_samples_leaf:
    - 1
    - 2
    - 5
    - 10
    criterion:
    - squared_error
    - absolute_error
    max_features:
    - sqrt
    - log2
    - null
    ccp_alpha:
    - 0.0
    - 0.0001
    - 0.001
    - 0.01
cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/lo/drd2/decision_tree/dt_maccs_drd2_lo.yaml
================================================================================

experiment:
  name: dt_maccs_drd2_lo
  task: lo
  dataset: drd2
fingerprint:
  type: maccs
model:
  name: dt
  fixed:
    random_state: 42
  search:
    max_depth:
    - 3
    - 5
    - 7
    - 10
    - 15
    - 20
    - null
    min_samples_split:
    - 2
    - 5
    - 10
    - 20
    min_samples_leaf:
    - 1
    - 2
    - 5
    - 10
    criterion:
    - squared_error
    - absolute_error
    max_features:
    - sqrt
    - log2
    - null
    ccp_alpha:
    - 0.0
    - 0.0001
    - 0.001
    - 0.01
cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/lo/drd2/dummy/dummy_drd2_lo.yaml
================================================================================

experiment:
  name: dummy_drd2_lo
  task: lo
  dataset: drd2

fingerprint:
  type: ecfp4        

model:
  name: dummy
  fixed:
    strategy: mean   
  search: {}

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/lo/drd2/gradient_boosting/gb_drd2_lo.yaml
================================================================================

experiment:
  name: gb_drd2_lo
  task: lo
  dataset: drd2

fingerprint:
  types: [maccs]

model:
  name: gb
  fixed:
    random_state: 42
  search:
    n_estimators: [100, 200, 300, 500]
    learning_rate: [0.01, 0.05, 0.1, 0.2]
    max_depth: [3, 4, 5, 6]
    min_samples_split: [2, 5, 10]
    min_samples_leaf: [1, 2, 4]
    subsample: [0.7, 0.8, 1.0]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: random
  n_iter: 80
  random_state: 42

================================================================================
FILE: configs/lo/drd2/knn/knn_drd2_lo.yaml
================================================================================

experiment:
  name: knn_ecfp4_drd2_lo
  task: lo
  dataset: drd2
fingerprint:
  types:
  - ecfp4
  - maccs
model:
  name: knn
  fixed:
    algorithm: auto
  search:
    n_neighbors:
    - 3
    - 5
    - 7
    - 11
    - 15
    - 21
    - 31
    - 51
    - 75
    - 101
    metric:
    - euclidean
    - minkowski
    weights:
    - uniform
    - distance
cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/lo/drd2/lr/lr_drd2_lo.yaml
================================================================================

experiment:
  name: lr_desc_drd2_lo
  task: lo
  dataset: drd2

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: linreg
  search:
    fit_intercept: [true, false]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/lo/drd2/random_forest/rf_drd2_lo.yaml
================================================================================

experiment:
  name: rf_drd2_lo
  task: lo
  dataset: drd2

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: rf
  fixed:
    random_state: 42
  search:
    n_estimators: [50, 100, 200, 300, 500, 800]
    max_depth: [null, 5, 10, 15, 20, 30, 50]
    min_samples_split: [2, 5, 10, 20]
    min_samples_leaf: [1, 2, 4, 8]
    max_features: [sqrt, log2, 0.1, 0.2, 0.3, 0.5]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: random
  n_iter: 100
  random_state: 42

================================================================================
FILE: configs/lo/drd2/svm/svr_linear_drd2_lo.yaml
================================================================================

experiment:
  name: svr_linear_drd2_lo
  task: lo
  dataset: drd2

fingerprint:
  types: [ecfp4, maccs]

model:
  name: svm
  fixed:
    kernel: linear
  search:
    C: [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    epsilon: [0.01, 0.05, 0.1, 0.2, 0.5]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/lo/drd2/svm/svr_poly_drd2_lo.yaml
================================================================================

experiment:
  name: svr_poly_drd2_lo
  task: lo
  dataset: drd2
fingerprint:
  types:
  - ecfp4
  - maccs
  - rdkit_desc
model:
  name: svm
  fixed:
    kernel: poly
  search:
    C:
    - 0.01
    - 0.1
    - 0.5
    - 1.0
    - 2.0
    - 5.0
    - 10.0
    - 50.0
    degree:
    - 2
    - 3
    - 4
    - 5
    coef0:
    - 0.0
    - 0.1
    - 0.5
    - 1.0
    gamma:
    - scale
    - auto
    epsilon:
    - 0.01
    - 0.05
    - 0.1
    - 0.2
cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: random
  random_state: 42
  n_iter: 50


================================================================================
FILE: configs/lo/drd2/svm/svr_rbf_drd2_lo.yaml
================================================================================

experiment:
  name: svr_rbf_drd2_lo
  task: lo
  dataset: drd2

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: svm
  fixed:
    kernel: rbf
  search:
    C: [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    gamma: [scale, auto, 0.0001, 0.001, 0.01, 0.1]
    epsilon: [0.01, 0.05, 0.1, 0.2]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/lo/drd2/svm/svr_tanimoto_drd2_lo.yaml
================================================================================

experiment:
  name: svr_tanimoto_drd2_lo
  task: lo
  dataset: drd2

fingerprint:
  types: [ecfp4, maccs]

model:
  name: svm
  fixed:
    kernel: tanimoto
  search:
    C: [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    epsilon: [0.01, 0.05, 0.1, 0.2]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/lo/drd2/xgboost/xgb_drd2_lo.yaml
================================================================================

experiment:
  name: xgb_drd2_lo
  task: lo
  dataset: drd2

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: xgb
  fixed:
    random_state: 42
    n_jobs: -1
    objective: reg:squarederror
  search:
    n_estimators: [100, 200, 300]
    learning_rate: [0.05, 0.1, 0.2]
    max_depth: [3, 4, 6]
    subsample: [0.8, 1.0]
    colsample_bytree: [0.8, 1.0]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: random
  n_iter: 60
  random_state: 42

================================================================================
FILE: configs/lo/kcnh2/decision_tree/dt_ecfp4_kcnh2_lo.yaml
================================================================================

experiment:
  name: dt_ecfp4_kcnh2_lo
  task: lo
  dataset: kcnh2
fingerprint:
  type: ecfp4
model:
  name: dt
  fixed:
    random_state: 42
  search:
    max_depth:
    - 3
    - 5
    - 7
    - 10
    - 15
    - 20
    - null
    min_samples_split:
    - 2
    - 5
    - 10
    - 20
    min_samples_leaf:
    - 1
    - 2
    - 5
    - 10
    criterion:
    - squared_error
    - absolute_error
    max_features:
    - sqrt
    - log2
    - null
    ccp_alpha:
    - 0.0
    - 0.0001
    - 0.001
    - 0.01
cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/lo/kcnh2/decision_tree/dt_maccs_kcnh2_lo.yaml
================================================================================

experiment:
  name: dt_maccs_kcnh2_lo
  task: lo
  dataset: kcnh2
fingerprint:
  type: maccs
model:
  name: dt
  fixed:
    random_state: 42
  search:
    max_depth:
    - 3
    - 5
    - 7
    - 10
    - 15
    - 20
    - null
    min_samples_split:
    - 2
    - 5
    - 10
    - 20
    min_samples_leaf:
    - 1
    - 2
    - 5
    - 10
    criterion:
    - squared_error
    - absolute_error
    max_features:
    - sqrt
    - log2
    - null
    ccp_alpha:
    - 0.0
    - 0.0001
    - 0.001
    - 0.01
cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/lo/kcnh2/dummy/dummy_kcnh2_lo.yaml
================================================================================

experiment:
  name: dummy_kcnh2_lo
  task: lo
  dataset: kcnh2

fingerprint:
  type: ecfp4        

model:
  name: dummy
  fixed:
    strategy: mean   
  search: {}

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/lo/kcnh2/gradient_boosting/gb_kcnh2_lo.yaml
================================================================================

experiment:
  name: gb_kcnh2_lo
  task: lo
  dataset: kcnh2

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: gb
  fixed:
    random_state: 42
  search:
    n_estimators: [100, 200, 300, 500]
    learning_rate: [0.01, 0.05, 0.1, 0.2]
    max_depth: [3, 4, 5, 6]
    min_samples_split: [2, 5, 10]
    min_samples_leaf: [1, 2, 4]
    subsample: [0.7, 0.8, 1.0]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: random
  n_iter: 50
  random_state: 42

================================================================================
FILE: configs/lo/kcnh2/knn/knn_kcnh2_lo.yaml
================================================================================

experiment:
  name: knn_ecfp4_kcnh2_lo
  task: lo
  dataset: kcnh2
fingerprint:
  types:
  - ecfp4
  - maccs
model:
  name: knn
  fixed:
    algorithm: auto
  search:
    n_neighbors:
    - 3
    - 5
    - 7
    - 11
    - 15
    - 21
    - 31
    - 51
    - 75
    - 101
    metric:
    - euclidean
    - minkowski
    weights:
    - uniform
    - distance
cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/lo/kcnh2/lr/lr_kcnh2_lo.yaml
================================================================================

experiment:
  name: lr_desc_kcnh2_lo
  task: lo
  dataset: kcnh2

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: linreg
  search:
    fit_intercept: [true, false]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/lo/kcnh2/random_forest/rf_kcnh2_lo.yaml
================================================================================

experiment:
  name: rf_kcnh2_lo
  task: lo
  dataset: kcnh2

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: rf
  fixed:
    random_state: 42
  search:
    n_estimators: [50, 100, 200, 300, 500, 800]
    max_depth: [null, 5, 10, 15, 20, 30, 50]
    min_samples_split: [2, 5, 10, 20]
    min_samples_leaf: [1, 2, 4, 8]
    max_features: [sqrt, log2, 0.1, 0.2, 0.3, 0.5]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: random
  n_iter: 100
  random_state: 42

================================================================================
FILE: configs/lo/kcnh2/svm/svr_linear_kcnh2_lo.yaml
================================================================================

experiment:
  name: svr_linear_kcnh2_lo
  task: lo
  dataset: kcnh2

fingerprint:
  types: [ecfp4, maccs]

model:
  name: svm
  fixed:
    kernel: linear
  search:
    C: [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    epsilon: [0.01, 0.05, 0.1, 0.2, 0.5]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/lo/kcnh2/svm/svr_poly_kcnh2_lo.yaml
================================================================================

experiment:
  name: svr_poly_kcnh2_lo
  task: lo
  dataset: kcnh2
fingerprint:
  types:
  - ecfp4
  - maccs
  - rdkit_desc
model:
  name: svm
  fixed:
    kernel: poly
  search:
    C:
    - 0.01
    - 0.1
    - 1.0
    - 5.0
    - 10.0
    degree:
    - 2
    - 3
    coef0:
    - 0.0
    - 0.5
    - 1.0
    gamma:
    - scale
    - auto
    epsilon:
    - 0.05
    - 0.1
    - 0.2
cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: random
  n_iter: 50
  random_state: 42


================================================================================
FILE: configs/lo/kcnh2/svm/svr_rbf_kcnh2_lo.yaml
================================================================================

experiment:
  name: svr_rbf_kcnh2_lo
  task: lo
  dataset: kcnh2

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: svm
  fixed:
    kernel: rbf
  search:
    C: [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    gamma: [scale, auto, 0.0001, 0.001, 0.01, 0.1]
    epsilon: [0.01, 0.05, 0.1, 0.2, 0.5]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/lo/kcnh2/svm/svr_tanimoto_kcnh2_lo.yaml
================================================================================

experiment:
  name: svr_tanimoto_kcnh2_lo
  task: lo
  dataset: kcnh2

fingerprint:
  types: [ecfp4, maccs]

model:
  name: svm
  fixed:
    kernel: tanimoto
  search:
    C: [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    epsilon: [0.01, 0.05, 0.1, 0.2, 0.5]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/lo/kcnh2/xgboost/xgb_kcnh2_lo.yaml
================================================================================

experiment:
  name: xgb_kcnh2_lo
  task: lo
  dataset: kcnh2

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: xgb
  fixed:
    random_state: 42
    n_jobs: -1
    objective: reg:squarederror
  search:
    n_estimators: [100, 200, 300]
    learning_rate: [0.05, 0.1, 0.2]
    max_depth: [3, 4, 6]
    subsample: [0.8, 1.0]
    colsample_bytree: [0.8, 1.0]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: random
  n_iter: 60
  random_state: 42

================================================================================
FILE: configs/lo/kdr/decision_tree/dt_ecfp4_kdr_lo.yaml
================================================================================

experiment:
  name: dt_ecfp4_kdr_lo
  task: lo
  dataset: kdr
fingerprint:
  type: ecfp4
model:
  name: dt
  fixed:
    random_state: 42
  search:
    max_depth:
    - 3
    - 5
    - 7
    - 10
    - 15
    - 20
    - null
    min_samples_split:
    - 2
    - 5
    - 10
    - 20
    min_samples_leaf:
    - 1
    - 2
    - 5
    - 10
    criterion:
    - squared_error
    - absolute_error
    max_features:
    - sqrt
    - log2
    - null
    ccp_alpha:
    - 0.0
    - 0.0001
    - 0.001
    - 0.01
cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/lo/kdr/decision_tree/dt_maccs_kdr_lo.yaml
================================================================================

experiment:
  name: dt_maccs_kdr_lo
  task: lo
  dataset: kdr
fingerprint:
  type: maccs
model:
  name: dt
  fixed:
    random_state: 42
  search:
    max_depth:
    - 3
    - 5
    - 7
    - 10
    - 15
    - 20
    - null
    min_samples_split:
    - 2
    - 5
    - 10
    - 20
    min_samples_leaf:
    - 1
    - 2
    - 5
    - 10
    criterion:
    - squared_error
    - absolute_error
    max_features:
    - sqrt
    - log2
    - null
    ccp_alpha:
    - 0.0
    - 0.0001
    - 0.001
    - 0.01
cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/lo/kdr/dummy/dummy_kdr_lo.yaml
================================================================================

experiment:
  name: dummy_kdr_lo
  task: lo
  dataset: kdr

fingerprint:
  type: ecfp4        

model:
  name: dummy
  fixed:
    strategy: mean  
  search: {}

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/lo/kdr/gradient_boosting/gb_kdr_lo.yaml
================================================================================

experiment:
  name: gb_kdr_lo
  task: lo
  dataset: kdr

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: gb
  fixed:
    random_state: 42
  search:
    n_estimators: [100, 200, 300, 500]
    learning_rate: [0.01, 0.05, 0.1, 0.2]
    max_depth: [3, 4, 5, 6]
    min_samples_split: [2, 5, 10]
    min_samples_leaf: [1, 2, 4]
    subsample: [0.7, 0.8, 1.0]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: random
  n_iter: 80
  random_state: 42

================================================================================
FILE: configs/lo/kdr/knn/knn_kdr_lo.yaml
================================================================================

experiment:
  name: knn_ecfp4_kdr_lo
  task: lo
  dataset: kdr
fingerprint:
  types:
  - ecfp4
  - maccs
model:
  name: knn
  fixed:
    algorithm: auto
  search:
    n_neighbors:
    - 3
    - 5
    - 7
    - 11
    - 15
    - 21
    - 31
    - 51
    - 75
    - 101
    metric:
    - euclidean
    - minkowski
    weights:
    - uniform
    - distance
cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42


================================================================================
FILE: configs/lo/kdr/lr/lr_kdr_lo.yaml
================================================================================

experiment:
  name: lr_ecfp4_kdr_lo
  task: lo
  dataset: kdr

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: linreg
  search:
    fit_intercept: [true, false]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/lo/kdr/random_forest/rf_kdr_lo.yaml
================================================================================

experiment:
  name: rf_kdr_lo
  task: lo
  dataset: kdr

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: rf
  fixed:
    random_state: 42
  search:
    n_estimators: [50, 100, 200, 300, 500, 800]
    max_depth: [null, 5, 10, 15, 20, 30, 50]
    min_samples_split: [2, 5, 10, 20]
    min_samples_leaf: [1, 2, 4, 8]
    max_features: [sqrt, log2, 0.1, 0.2, 0.3, 0.5]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: random
  n_iter: 100
  random_state: 42

================================================================================
FILE: configs/lo/kdr/svm/svr_linear_kdr_lo.yaml
================================================================================

experiment:
  name: svr_linear_kdr_lo
  task: lo
  dataset: kdr

fingerprint:
  types: [ecfp4, maccs]

model:
  name: svm
  fixed:
    kernel: linear
  search:
    C: [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    epsilon: [0.01, 0.05, 0.1, 0.2, 0.5]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/lo/kdr/svm/svr_poly_kdr_lo.yaml
================================================================================

experiment:
  name: svr_poly_kdr_lo
  task: lo
  dataset: kdr
fingerprint:
  types:
  - ecfp4
  - maccs
  - rdkit_desc
model:
  name: svm
  fixed:
    kernel: poly
  search:
    C:
    - 0.01
    - 0.1
    - 0.5
    - 1.0
    - 2.0
    - 5.0
    - 10.0
    - 50.0
    degree:
    - 2
    - 3
    - 4
    - 5
    coef0:
    - 0.0
    - 0.1
    - 0.5
    - 1.0
    gamma:
    - scale
    - auto
    epsilon:
    - 0.01
    - 0.05
    - 0.1
    - 0.2
cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: random
  random_state: 42
  n_iter: 50


================================================================================
FILE: configs/lo/kdr/svm/svr_rbf_kdr_lo.yaml
================================================================================

experiment:
  name: svr_rbf_kdr_lo
  task: lo
  dataset: kdr

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: svm
  fixed:
    kernel: rbf
  search:
    C: [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    gamma: [scale, auto, 0.0001, 0.001, 0.01, 0.1]
    epsilon: [0.01, 0.05, 0.1, 0.2]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/lo/kdr/svm/svr_tanimoto_kdr_lo.yaml
================================================================================

experiment:
  name: svr_tanimoto_kdr_lo
  task: lo
  dataset: kdr

fingerprint:
  types: [ecfp4, maccs]

model:
  name: svm
  fixed:
    kernel: tanimoto
  search:
    C: [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    epsilon: [0.01, 0.05, 0.1, 0.2]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: grid
  random_state: 42

================================================================================
FILE: configs/lo/kdr/xgboost/xgb_kdr_lo.yaml
================================================================================

experiment:
  name: xgb_kdr_lo
  task: lo
  dataset: kdr

fingerprint:
  types: [ecfp4, maccs, rdkit_desc]

model:
  name: xgb
  fixed:
    random_state: 42
    n_jobs: -1
    objective: reg:squarederror
  search:
    n_estimators: [100, 200, 300]
    learning_rate: [0.05, 0.1, 0.2]
    max_depth: [3, 4, 6]
    subsample: [0.8, 1.0]
    colsample_bytree: [0.8, 1.0]

cv:
  inner_k: 2
  scoring: neg_mean_absolute_error
  search_strategy: random
  n_iter: 60
  random_state: 42
