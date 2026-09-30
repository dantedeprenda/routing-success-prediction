# Evidence: twinrouterbench_test

## A. Few-shot discrimination (within-category AUROC, P(cheapest suffices))

Hierarchical paired bootstrap (2000 replicates: 5 label seeds x stratified test set).

| k | router | baseline | router | baseline | diff | 95% CI | seeds won |
|---|---|---|---|---|---|---|---|
| 25 | jev-4k_features | tfidf_logreg | 0.654 | 0.563 | +0.091 | [-0.045, +0.188] | 5/5 |
| 25 | jev-4k_features | knn10 | 0.654 | 0.497 | +0.157 | [+0.033, +0.247] | 5/5 |
| 25 | jev-4k_features | bge_logreg | 0.654 | 0.533 | +0.121 | [-0.043, +0.233] | 5/5 |
| 25 | jev-4k_features | bge_knn | 0.654 | 0.516 | +0.138 | [-0.001, +0.235] | 5/5 |
| 25 | jev-4k_features | bgeL_logreg | 0.654 | 0.556 | +0.097 | [-0.027, +0.214] | 5/5 |
| 25 | jev-4k_features | bgeL_knn | 0.654 | 0.517 | +0.137 | [+0.008, +0.253] | 5/5 |
| 50 | jev-4k_features | tfidf_logreg | 0.669 | 0.556 | +0.112 | [-0.046, +0.211] | 5/5 |
| 50 | jev-4k_features | knn10 | 0.669 | 0.499 | +0.170 | [+0.028, +0.255] | 5/5 |
| 50 | jev-4k_features | bge_logreg | 0.669 | 0.533 | +0.136 | [-0.025, +0.252] | 5/5 |
| 50 | jev-4k_features | bge_knn | 0.669 | 0.580 | +0.089 | [-0.034, +0.189] | 5/5 |
| 50 | jev-4k_features | bgeL_logreg | 0.669 | 0.554 | +0.115 | [-0.014, +0.227] | 5/5 |
| 50 | jev-4k_features | bgeL_knn | 0.669 | 0.540 | +0.129 | [+0.009, +0.222] | 5/5 |
| 100 | jev-4k_features | tfidf_logreg | 0.675 | 0.575 | +0.100 | [-0.055, +0.194] | 5/5 |
| 100 | jev-4k_features | knn10 | 0.675 | 0.528 | +0.148 | [+0.010, +0.242] | 5/5 |
| 100 | jev-4k_features | bge_logreg | 0.675 | 0.584 | +0.091 | [-0.049, +0.200] | 5/5 |
| 100 | jev-4k_features | bge_knn | 0.675 | 0.604 | +0.071 | [-0.051, +0.179] | 5/5 |
| 100 | jev-4k_features | bgeL_logreg | 0.675 | 0.580 | +0.095 | [-0.026, +0.202] | 5/5 |
| 100 | jev-4k_features | bgeL_knn | 0.675 | 0.572 | +0.103 | [-0.015, +0.200] | 5/5 |
| all | jev-4k_features | tfidf_logreg | 0.718 | 0.722 | -0.004 | [-0.149, +0.093] | 0/5 |
| all | jev-4k_features | knn10 | 0.718 | 0.598 | +0.120 | [-0.029, +0.226] | 5/5 |
| all | jev-4k_features | bge_logreg | 0.718 | 0.656 | +0.062 | [-0.064, +0.166] | 5/5 |
| all | jev-4k_features | bge_knn | 0.718 | 0.687 | +0.031 | [-0.073, +0.135] | 5/5 |
| all | jev-4k_features | bgeL_logreg | 0.718 | 0.591 | +0.127 | [-0.004, +0.242] | 5/5 |
| all | jev-4k_features | bgeL_knn | 0.718 | 0.714 | +0.004 | [-0.109, +0.118] | 5/5 |

## B. Operating points vs always-mid_high (0.828 @ $0.00273), all labels

| router | op | quality | cost | d quality [95% CI] | d cost [95% CI] | McNemar p |
|---|---|---|---|---|---|---|
| jev-4k_choice | budget_mid | 0.813 | $0.00277 | -0.015 [-0.030, -0.002] | +0.00004 [-0.00038, +0.00056] | 0.065 |
| jev-4k_choice | match_frac | 0.870 | $0.00651 | +0.043 [+0.026, +0.062] | +0.00378 [+0.00274, +0.00490] | 0.000 |
| jev-4k_features | budget_mid | 0.751 | $0.00217 | -0.077 [-0.109, -0.047] | -0.00056 [-0.00107, +0.00003] | 0.000 |
| jev-4k_features | match_frac | 0.891 | $0.00794 | +0.064 [+0.028, +0.098] | +0.00521 [+0.00387, +0.00651] | 0.001 |
| tfidf_logreg | budget_mid | 0.783 | $0.00190 | -0.045 [-0.068, -0.023] | -0.00083 [-0.00109, -0.00053] | 0.000 |
| tfidf_logreg | match_frac | 0.921 | $0.00884 | +0.094 [+0.055, +0.130] | +0.00611 [+0.00478, +0.00743] | 0.000 |
| bge_logreg | budget_mid | 0.743 | $0.00118 | -0.085 [-0.111, -0.062] | -0.00155 [-0.00162, -0.00148] | 0.000 |
| bge_logreg | match_frac | 0.834 | $0.00504 | +0.006 [-0.028, +0.038] | +0.00231 [+0.00129, +0.00341] | 0.791 |
| category_best | budget_mid | 0.789 | $0.00170 | -0.038 [-0.057, -0.023] | -0.00103 [-0.00112, -0.00095] | 0.000 |
| category_best | match_frac | 0.962 | $0.01218 | +0.134 [+0.096, +0.172] | +0.00945 [+0.00789, +0.01096] | 0.000 |
