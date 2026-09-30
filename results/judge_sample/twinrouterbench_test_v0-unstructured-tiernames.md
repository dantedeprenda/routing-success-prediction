# Judge sample: twinrouterbench_test

470 stratified test examples (seed 0; ids in `twinrouterbench_test_ids.txt`). Decision routers use the thresholds they picked on labeled train data (raw variant); judges and cascades need none. Cost = model + router.

| router | quality | cost | router $/call | latency p50 / p95 s | AUROC cheap-ok | Jev-curve quality at this cost | notes |
|---|---|---|---|---|---|---|---|
| always:low | 0.711 | $0.00092 | $0.00000 | 0.00 / 0.00 |  | 0.000 |  |
| always:mid | 0.772 | $0.00140 | $0.00000 | 0.00 / 0.00 |  | 0.765 |  |
| always:mid_high | 0.828 | $0.00273 | $0.00000 | 0.00 / 0.00 |  | 0.820 |  |
| always:high | 1.000 | $0.02530 | $0.00000 | 0.00 / 0.00 |  | 1.000 |  |
| judge-sonnet | 0.768 | $0.00615 | $0.00417 | 2.50 / 5.24 |  | 0.868 | unparsed 0, errors 0, new spend $0.66 |
| judge-luna | 0.768 | $0.00291 | $0.00020 | 2.61 / 5.39 |  | 0.825 | unparsed 0, errors 0, new spend $0.01 |
| category_best@budget_mid | 0.789 | $0.00170 | $0.00000 | 0.00 / 0.00 | 0.86 [0.83, 0.90] | 0.784 |  |
| category_best@match_frac | 0.962 | $0.01218 | $0.00000 | 0.00 / 0.00 | 0.86 [0.83, 0.90] | 0.938 |  |
| jev-4k_choice@budget_mid | 0.813 | $0.00277 | $0.00009 | 0.32 / 0.50 | 0.74 [0.69, 0.79] |  |  |
| jev-4k_choice@match_frac | 0.870 | $0.00651 | $0.00009 | 0.32 / 0.50 | 0.74 [0.69, 0.79] |  |  |
| jev-4k_difficulty@budget_mid | 0.802 | $0.00227 | $0.00009 | 0.32 / 0.50 | 0.72 [0.67, 0.76] | 0.809 |  |
| jev-4k_difficulty@match_frac | 0.870 | $0.00567 | $0.00009 | 0.32 / 0.50 | 0.72 [0.67, 0.76] | 0.862 |  |
| jev-4k_features@budget_mid | 0.751 | $0.00217 | $0.00009 | 0.32 / 0.50 | 0.88 [0.83, 0.91] | 0.806 |  |
| jev-4k_features@match_frac | 0.891 | $0.00794 | $0.00009 | 0.32 / 0.50 | 0.88 [0.83, 0.91] | 0.892 |  |
| jev-4k_tier_nouls@budget_mid | 0.740 | $0.00196 | $0.00009 | 0.32 / 0.50 | 0.76 [0.71, 0.80] | 0.798 |  |
| jev-4k_tier_nouls@match_frac | 0.879 | $0.00910 | $0.00009 | 0.32 / 0.50 | 0.76 [0.71, 0.80] | 0.908 |  |
| jev_choice@budget_mid | 0.826 | $0.00303 | $0.00014 | 0.30 / 0.50 | 0.76 [0.72, 0.80] | 0.827 |  |
| jev_choice@match_frac | 0.883 | $0.00918 | $0.00014 | 0.30 / 0.50 | 0.76 [0.72, 0.80] | 0.909 |  |
| jev_difficulty@budget_mid | 0.800 | $0.00212 | $0.00014 | 0.30 / 0.50 | 0.73 [0.68, 0.77] | 0.804 |  |
| jev_difficulty@match_frac | 0.862 | $0.00523 | $0.00014 | 0.30 / 0.50 | 0.73 [0.68, 0.77] | 0.856 |  |
| jev_features@budget_mid | 0.745 | $0.00193 | $0.00014 | 0.30 / 0.50 | 0.88 [0.84, 0.91] | 0.796 |  |
| jev_features@match_frac | 0.894 | $0.00857 | $0.00014 | 0.30 / 0.50 | 0.88 [0.84, 0.91] | 0.901 |  |
| jev_tier_nouls@budget_mid | 0.732 | $0.00247 | $0.00014 | 0.30 / 0.50 | 0.71 [0.66, 0.76] | 0.814 |  |
| jev_tier_nouls@match_frac | 0.889 | $0.01072 | $0.00014 | 0.30 / 0.50 | 0.71 [0.66, 0.76] | 0.924 |  |
| knn10@budget_mid | 0.713 | $0.00095 | $0.00000 | 0.00 / 0.00 | 0.89 [0.85, 0.92] | 0.000 |  |
| knn10@match_frac | 0.787 | $0.00313 | $0.00000 | 0.00 / 0.00 | 0.89 [0.85, 0.92] | 0.828 |  |
| laya-ml_choice@budget_mid | 0.823 | $0.00238 | $0.00000 | 3.91 / 5.05 | 0.69 [0.64, 0.74] | 0.811 |  |
| laya-ml_choice@match_frac | 0.862 | $0.00506 | $0.00000 | 3.91 / 5.05 | 0.69 [0.64, 0.74] | 0.854 |  |
| laya-ml_difficulty@budget_mid | 0.800 | $0.00228 | $0.00000 | 3.91 / 5.05 | 0.44 [0.39, 0.50] | 0.809 |  |
| laya-ml_difficulty@match_frac | 0.879 | $0.00753 | $0.00000 | 3.91 / 5.05 | 0.44 [0.39, 0.50] | 0.887 |  |
| laya-ml_features@budget_mid | 0.787 | $0.00249 | $0.00000 | 3.91 / 5.05 | 0.69 [0.63, 0.73] | 0.814 |  |
| laya-ml_features@match_frac | 0.898 | $0.01200 | $0.00000 | 3.91 / 5.05 | 0.69 [0.63, 0.73] | 0.936 |  |
| laya-ml_tier_nouls@budget_mid | 0.717 | $0.00182 | $0.00000 | 3.91 / 5.05 | 0.51 [0.46, 0.56] | 0.790 |  |
| laya-ml_tier_nouls@match_frac | 0.881 | $0.01448 | $0.00000 | 3.91 / 5.05 | 0.51 [0.46, 0.56] | 0.960 |  |
| tfidf_logreg@budget_mid | 0.783 | $0.00190 | $0.00000 | 0.00 / 0.00 | 0.89 [0.85, 0.92] | 0.795 |  |
| tfidf_logreg@match_frac | 0.921 | $0.00884 | $0.00000 | 0.00 / 0.00 | 0.89 [0.85, 0.92] | 0.904 |  |
| jev-4k_choice>judge-sonnet@0.3 | 0.815 | $0.00499 | $0.00095 | 0.34 / 2.86 |  |  | deferred 21% |
| jev-4k_choice>judge-sonnet@0.5 | 0.806 | $0.00517 | $0.00179 | 0.39 / 2.93 |  |  | deferred 41% |
| jev-4k_choice>judge-sonnet@0.7 | 0.781 | $0.00555 | $0.00291 | 2.80 / 2.95 |  |  | deferred 68% |
| jev-4k_choice>judge-sonnet@0.9 | 0.768 | $0.00597 | $0.00385 | 2.81 / 2.97 |  |  | deferred 90% |
| jev-4k_choice>judge-luna@0.3 | 0.817 | $0.00420 | $0.00013 | 0.34 / 2.97 |  |  | deferred 21% |
| jev-4k_choice>judge-luna@0.5 | 0.806 | $0.00402 | $0.00017 | 0.39 / 3.04 |  |  | deferred 41% |
| jev-4k_choice>judge-luna@0.7 | 0.783 | $0.00359 | $0.00022 | 2.91 / 3.07 |  |  | deferred 68% |
| jev-4k_choice>judge-luna@0.9 | 0.770 | $0.00312 | $0.00026 | 2.92 / 3.09 |  |  | deferred 90% |

Paired bootstrap vs jev-4k_choice@match_frac (mean diff, 95% CI):

- judge-sonnet: quality -0.102 [-0.130, -0.077], cost -0.00036 [-0.00148, +0.00067]
- judge-luna: quality -0.102 [-0.130, -0.077], cost -0.00360 [-0.00483, -0.00238]

Tier metrics (RowPass / TrajPass):

- judge-sonnet: 0.768 / 0.630
- judge-luna: 0.768 / 0.615
- jev-4k_choice@match_frac: 0.870 / 0.657
