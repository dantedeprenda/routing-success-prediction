# Evidence: llmrouterbench_test

## A. Few-shot discrimination (within-category AUROC, P(cheapest suffices))

Hierarchical paired bootstrap (2000 replicates: 5 label seeds x stratified test set).

| k | router | baseline | router | baseline | diff | 95% CI | seeds won |
|---|---|---|---|---|---|---|---|
| 25 | jev_features | tfidf_logreg | 0.729 | 0.526 | +0.203 | [+0.156, +0.248] | 5/5 |
| 25 | jev_features | knn10 | 0.729 | 0.523 | +0.207 | [+0.166, +0.245] | 5/5 |
| 25 | jev_features | bge_logreg | 0.729 | 0.532 | +0.198 | [+0.155, +0.240] | 5/5 |
| 25 | jev_features | bge_knn | 0.729 | 0.542 | +0.188 | [+0.141, +0.233] | 5/5 |
| 25 | jev_features | bgeL_logreg | 0.729 | 0.516 | +0.213 | [+0.168, +0.256] | 5/5 |
| 25 | jev_features | bgeL_knn | 0.729 | 0.534 | +0.195 | [+0.151, +0.238] | 5/5 |
| 50 | jev_features | tfidf_logreg | 0.721 | 0.540 | +0.182 | [+0.138, +0.226] | 5/5 |
| 50 | jev_features | knn10 | 0.721 | 0.526 | +0.195 | [+0.146, +0.241] | 5/5 |
| 50 | jev_features | bge_logreg | 0.721 | 0.555 | +0.166 | [+0.121, +0.213] | 5/5 |
| 50 | jev_features | bge_knn | 0.721 | 0.531 | +0.190 | [+0.147, +0.232] | 5/5 |
| 50 | jev_features | bgeL_logreg | 0.721 | 0.546 | +0.175 | [+0.121, +0.230] | 5/5 |
| 50 | jev_features | bgeL_knn | 0.721 | 0.518 | +0.203 | [+0.158, +0.249] | 5/5 |
| 100 | jev_features | tfidf_logreg | 0.749 | 0.589 | +0.160 | [+0.114, +0.206] | 5/5 |
| 100 | jev_features | knn10 | 0.749 | 0.575 | +0.174 | [+0.133, +0.214] | 5/5 |
| 100 | jev_features | bge_logreg | 0.749 | 0.594 | +0.155 | [+0.109, +0.204] | 5/5 |
| 100 | jev_features | bge_knn | 0.749 | 0.562 | +0.187 | [+0.144, +0.231] | 5/5 |
| 100 | jev_features | bgeL_logreg | 0.749 | 0.588 | +0.161 | [+0.115, +0.210] | 5/5 |
| 100 | jev_features | bgeL_knn | 0.749 | 0.551 | +0.198 | [+0.149, +0.251] | 5/5 |
| 200 | jev_features | tfidf_logreg | 0.744 | 0.580 | +0.164 | [+0.121, +0.209] | 5/5 |
| 200 | jev_features | knn10 | 0.744 | 0.563 | +0.181 | [+0.135, +0.226] | 5/5 |
| 200 | jev_features | bge_logreg | 0.744 | 0.575 | +0.169 | [+0.131, +0.207] | 5/5 |
| 200 | jev_features | bge_knn | 0.744 | 0.559 | +0.185 | [+0.142, +0.229] | 5/5 |
| 200 | jev_features | bgeL_logreg | 0.744 | 0.552 | +0.192 | [+0.155, +0.231] | 5/5 |
| 200 | jev_features | bgeL_knn | 0.744 | 0.545 | +0.199 | [+0.157, +0.240] | 5/5 |
| all | jev_features | tfidf_logreg | 0.757 | 0.664 | +0.092 | [+0.054, +0.133] | 5/5 |
| all | jev_features | knn10 | 0.757 | 0.621 | +0.135 | [+0.093, +0.178] | 5/5 |
| all | jev_features | bge_logreg | 0.757 | 0.648 | +0.109 | [+0.071, +0.147] | 5/5 |
| all | jev_features | bge_knn | 0.757 | 0.651 | +0.105 | [+0.061, +0.151] | 5/5 |
| all | jev_features | bgeL_logreg | 0.757 | 0.650 | +0.106 | [+0.068, +0.147] | 5/5 |
| all | jev_features | bgeL_knn | 0.757 | 0.645 | +0.111 | [+0.069, +0.151] | 5/5 |

## B. Operating points vs always-deepseek-r1-0528 (0.785 @ $0.01449), all labels

| router | op | quality | cost | d quality [95% CI] | d cost [95% CI] | McNemar p |
|---|---|---|---|---|---|---|
| jev_choice | budget_mid | 0.788 | $0.01321 | +0.003 [-0.008, +0.015] | -0.00128 [-0.00245, +0.00013] | 0.699 |
| jev_choice | match_frac | 0.790 | $0.01420 | +0.005 [-0.005, +0.016] | -0.00029 [-0.00159, +0.00117] | 0.419 |
| jev_features | budget_mid | 0.772 | $0.01404 | -0.013 [-0.030, +0.005] | -0.00046 [-0.00203, +0.00130] | 0.171 |
| jev_features | match_frac | 0.794 | $0.01758 | +0.009 [-0.008, +0.027] | +0.00308 [+0.00142, +0.00492] | 0.346 |
| tfidf_logreg | budget_mid | 0.762 | $0.01132 | -0.023 [-0.039, -0.006] | -0.00317 [-0.00453, -0.00170] | 0.011 |
| tfidf_logreg | match_frac | 0.748 | $0.00849 | -0.036 [-0.053, -0.018] | -0.00600 [-0.00718, -0.00468] | 0.000 |
| bge_logreg | budget_mid | 0.772 | $0.01279 | -0.013 [-0.030, +0.004] | -0.00171 [-0.00316, -0.00007] | 0.155 |
| bge_logreg | match_frac | 0.783 | $0.01620 | -0.002 [-0.019, +0.015] | +0.00171 [+0.00007, +0.00357] | 0.928 |
| category_best | budget_mid | 0.778 | $0.00864 | -0.007 [-0.020, +0.006] | -0.00585 [-0.00631, -0.00541] | 0.356 |
| category_best | match_frac | 0.807 | $0.02508 | +0.023 [+0.006, +0.040] | +0.01059 [+0.00887, +0.01253] | 0.013 |

## C. Leave-one-dataset-out (AUROC on the held-out dataset)

Heads trained on the other datasets' train split (k=25: mean ± sd over 5 seeds);
scored on every non-design prompt of the held-out dataset.

| held out | n (neg) | k | router | AUROC | sd | 95% CI (one seed) |
|---|---|---|---|---|---|---|
| aime | 21 (11) | 25 | knn10 | 0.466 | 0.067 | [0.29, 0.72] |
| aime | 21 (11) | 25 | tfidf_logreg | 0.380 | 0.063 | [0.05, 0.59] |
| aime | 21 (11) | 25 | bge_logreg | 0.502 | 0.049 | [0.22, 0.74] |
| aime | 21 (11) | 25 | bge_knn | 0.485 | 0.066 | [0.50, 0.65] |
| aime | 21 (11) | 25 | bgeL_logreg | 0.553 | 0.035 | [0.34, 0.86] |
| aime | 21 (11) | 25 | bgeL_knn | 0.525 | 0.055 | [0.36, 0.62] |
| aime | 21 (11) | 25 | jev:features | 0.782 | 0.033 | [0.51, 0.96] |
| aime | 21 (11) | 25 | jev:choice | 0.682 | 0.000 | [0.39, 0.92] |
| aime | 21 (11) | 25 | jev:difficulty | 0.605 | 0.000 | [0.43, 0.78] |
| aime | 21 (11) | all | knn10 | 0.582 | 0.000 | [0.32, 0.82] |
| aime | 21 (11) | all | tfidf_logreg | 0.618 | 0.000 | [0.35, 0.86] |
| aime | 21 (11) | all | bge_logreg | 0.473 | 0.000 | [0.19, 0.76] |
| aime | 21 (11) | all | bge_knn | 0.514 | 0.000 | [0.26, 0.75] |
| aime | 21 (11) | all | bgeL_logreg | 0.464 | 0.000 | [0.21, 0.74] |
| aime | 21 (11) | all | bgeL_knn | 0.523 | 0.000 | [0.23, 0.79] |
| aime | 21 (11) | all | jev:features | 0.782 | 0.000 | [0.51, 0.97] |
| aime | 21 (11) | all | jev:choice | 0.682 | 0.000 | [0.39, 0.92] |
| aime | 21 (11) | all | jev:difficulty | 0.605 | 0.000 | [0.43, 0.78] |
| gpqa | 93 (24) | 25 | knn10 | 0.514 | 0.030 | [0.35, 0.59] |
| gpqa | 93 (24) | 25 | tfidf_logreg | 0.532 | 0.062 | [0.30, 0.58] |
| gpqa | 93 (24) | 25 | bge_logreg | 0.506 | 0.103 | [0.19, 0.46] |
| gpqa | 93 (24) | 25 | bge_knn | 0.515 | 0.087 | [0.25, 0.50] |
| gpqa | 93 (24) | 25 | bgeL_logreg | 0.526 | 0.113 | [0.22, 0.49] |
| gpqa | 93 (24) | 25 | bgeL_knn | 0.516 | 0.048 | [0.32, 0.58] |
| gpqa | 93 (24) | 25 | jev:features | 0.620 | 0.055 | [0.38, 0.66] |
| gpqa | 93 (24) | 25 | jev:choice | 0.562 | 0.000 | [0.43, 0.70] |
| gpqa | 93 (24) | 25 | jev:difficulty | 0.587 | 0.000 | [0.47, 0.70] |
| gpqa | 93 (24) | all | knn10 | 0.429 | 0.000 | [0.30, 0.56] |
| gpqa | 93 (24) | all | tfidf_logreg | 0.634 | 0.000 | [0.51, 0.76] |
| gpqa | 93 (24) | all | bge_logreg | 0.417 | 0.000 | [0.28, 0.54] |
| gpqa | 93 (24) | all | bge_knn | 0.461 | 0.000 | [0.32, 0.60] |
| gpqa | 93 (24) | all | bgeL_logreg | 0.528 | 0.000 | [0.39, 0.65] |
| gpqa | 93 (24) | all | bgeL_knn | 0.484 | 0.000 | [0.35, 0.62] |
| gpqa | 93 (24) | all | jev:features | 0.611 | 0.000 | [0.50, 0.73] |
| gpqa | 93 (24) | all | jev:choice | 0.562 | 0.000 | [0.43, 0.70] |
| gpqa | 93 (24) | all | jev:difficulty | 0.587 | 0.000 | [0.47, 0.70] |
| livecodebench | 949 (313) | 25 | knn10 | 0.514 | 0.055 | [0.45, 0.52] |
| livecodebench | 949 (313) | 25 | tfidf_logreg | 0.537 | 0.038 | [0.46, 0.54] |
| livecodebench | 949 (313) | 25 | bge_logreg | 0.516 | 0.043 | [0.43, 0.51] |
| livecodebench | 949 (313) | 25 | bge_knn | 0.495 | 0.027 | [0.42, 0.49] |
| livecodebench | 949 (313) | 25 | bgeL_logreg | 0.508 | 0.058 | [0.38, 0.46] |
| livecodebench | 949 (313) | 25 | bgeL_knn | 0.499 | 0.027 | [0.47, 0.53] |
| livecodebench | 949 (313) | 25 | jev:features | 0.620 | 0.180 | [0.33, 0.40] |
| livecodebench | 949 (313) | 25 | jev:choice | 0.813 | 0.000 | [0.79, 0.84] |
| livecodebench | 949 (313) | 25 | jev:difficulty | 0.742 | 0.000 | [0.71, 0.77] |
| livecodebench | 949 (313) | all | knn10 | 0.463 | 0.000 | [0.42, 0.50] |
| livecodebench | 949 (313) | all | tfidf_logreg | 0.495 | 0.000 | [0.46, 0.54] |
| livecodebench | 949 (313) | all | bge_logreg | 0.558 | 0.000 | [0.52, 0.60] |
| livecodebench | 949 (313) | all | bge_knn | 0.490 | 0.000 | [0.45, 0.53] |
| livecodebench | 949 (313) | all | bgeL_logreg | 0.496 | 0.000 | [0.46, 0.53] |
| livecodebench | 949 (313) | all | bgeL_knn | 0.507 | 0.000 | [0.47, 0.54] |
| livecodebench | 949 (313) | all | jev:features | 0.813 | 0.000 | [0.79, 0.84] |
| livecodebench | 949 (313) | all | jev:choice | 0.813 | 0.000 | [0.79, 0.84] |
| livecodebench | 949 (313) | all | jev:difficulty | 0.742 | 0.000 | [0.71, 0.77] |
| mmlupro | 2891 (469) | 25 | knn10 | 0.514 | 0.040 | [0.53, 0.58] |
| mmlupro | 2891 (469) | 25 | tfidf_logreg | 0.518 | 0.045 | [0.55, 0.61] |
| mmlupro | 2891 (469) | 25 | bge_logreg | 0.495 | 0.017 | [0.45, 0.50] |
| mmlupro | 2891 (469) | 25 | bge_knn | 0.484 | 0.019 | [0.44, 0.50] |
| mmlupro | 2891 (469) | 25 | bgeL_logreg | 0.499 | 0.040 | [0.41, 0.46] |
| mmlupro | 2891 (469) | 25 | bgeL_knn | 0.487 | 0.025 | [0.45, 0.50] |
| mmlupro | 2891 (469) | 25 | jev:features | 0.654 | 0.092 | [0.46, 0.51] |
| mmlupro | 2891 (469) | 25 | jev:choice | 0.664 | 0.000 | [0.64, 0.69] |
| mmlupro | 2891 (469) | 25 | jev:difficulty | 0.630 | 0.000 | [0.60, 0.66] |
| mmlupro | 2891 (469) | all | knn10 | 0.530 | 0.000 | [0.50, 0.56] |
| mmlupro | 2891 (469) | all | tfidf_logreg | 0.557 | 0.000 | [0.53, 0.58] |
| mmlupro | 2891 (469) | all | bge_logreg | 0.537 | 0.000 | [0.51, 0.56] |
| mmlupro | 2891 (469) | all | bge_knn | 0.524 | 0.000 | [0.49, 0.55] |
| mmlupro | 2891 (469) | all | bgeL_logreg | 0.577 | 0.000 | [0.55, 0.61] |
| mmlupro | 2891 (469) | all | bgeL_knn | 0.509 | 0.000 | [0.48, 0.54] |
| mmlupro | 2891 (469) | all | jev:features | 0.711 | 0.000 | [0.69, 0.73] |
| mmlupro | 2891 (469) | all | jev:choice | 0.664 | 0.000 | [0.64, 0.69] |
| mmlupro | 2891 (469) | all | jev:difficulty | 0.630 | 0.000 | [0.60, 0.66] |
| swe-bench | 395 (292) | 25 | knn10 | 0.511 | 0.013 | [0.47, 0.58] |
| swe-bench | 395 (292) | 25 | tfidf_logreg | 0.515 | 0.022 | [0.48, 0.61] |
| swe-bench | 395 (292) | 25 | bge_logreg | 0.512 | 0.043 | [0.41, 0.54] |
| swe-bench | 395 (292) | 25 | bge_knn | 0.519 | 0.020 | [0.49, 0.60] |
| swe-bench | 395 (292) | 25 | bgeL_logreg | 0.513 | 0.031 | [0.41, 0.54] |
| swe-bench | 395 (292) | 25 | bgeL_knn | 0.535 | 0.032 | [0.48, 0.55] |
| swe-bench | 395 (292) | 25 | jev:features | 0.601 | 0.035 | [0.47, 0.60] |
| swe-bench | 395 (292) | 25 | jev:choice | 0.591 | 0.000 | [0.53, 0.65] |
| swe-bench | 395 (292) | 25 | jev:difficulty | 0.638 | 0.000 | [0.58, 0.70] |
| swe-bench | 395 (292) | all | knn10 | 0.535 | 0.000 | [0.47, 0.60] |
| swe-bench | 395 (292) | all | tfidf_logreg | 0.490 | 0.000 | [0.43, 0.56] |
| swe-bench | 395 (292) | all | bge_logreg | 0.561 | 0.000 | [0.49, 0.62] |
| swe-bench | 395 (292) | all | bge_knn | 0.545 | 0.000 | [0.48, 0.60] |
| swe-bench | 395 (292) | all | bgeL_logreg | 0.536 | 0.000 | [0.47, 0.60] |
| swe-bench | 395 (292) | all | bgeL_knn | 0.548 | 0.000 | [0.48, 0.62] |
| swe-bench | 395 (292) | all | jev:features | 0.636 | 0.000 | [0.58, 0.70] |
| swe-bench | 395 (292) | all | jev:choice | 0.591 | 0.000 | [0.53, 0.65] |
| swe-bench | 395 (292) | all | jev:difficulty | 0.638 | 0.000 | [0.58, 0.70] |
