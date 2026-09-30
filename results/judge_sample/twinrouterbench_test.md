# Judge sample: twinrouterbench_test

470 stratified test examples (seed 0; ids in `twinrouterbench_test_ids.txt`). Decision routers use the thresholds they picked on labeled train data (raw variant); judges and cascades need none. Cost = model + router.
Judges saw model ids: deepseek-v3.2, minimax-m2.7, gemini-3-flash, claude-opus-4.6.

| router | quality | cost | router $/call | latency p50 / p95 s | AUROC cheap-ok | Jev-curve quality at this cost | notes |
|---|---|---|---|---|---|---|---|
| always:low | 0.711 | $0.00092 | $0.00000 | 0.00 / 0.00 |  | 0.000 |  |
| always:mid | 0.772 | $0.00140 | $0.00000 | 0.00 / 0.00 |  | 0.765 |  |
| always:mid_high | 0.828 | $0.00273 | $0.00000 | 0.00 / 0.00 |  | 0.820 |  |
| always:high | 1.000 | $0.02530 | $0.00000 | 0.00 / 0.00 |  | 1.000 |  |
| judge-sonnet | 0.874 | $0.01181 | $0.00496 | 2.78 / 5.08 |  | 0.934 | unparsed 0, errors 0, timeouts >9s 0, new spend $0.00 |
| judge-luna | 0.774 | $0.00305 | $0.00022 | 2.69 / 5.11 |  | 0.827 | unparsed 0, errors 0, timeouts >9s 0, new spend $0.00 |
| bgeL_knn@budget_mid | 0.732 | $0.00146 | $0.00000 | 0.00 / 0.00 | 0.89 [0.86, 0.93] | 0.770 |  |
| bgeL_knn@match_frac | 0.879 | $0.00783 | $0.00000 | 0.00 / 0.00 | 0.89 [0.86, 0.93] | 0.891 |  |
| bgeL_logreg@budget_mid | 0.828 | $0.00266 | $0.00000 | 0.00 / 0.00 | 0.87 [0.83, 0.91] | 0.818 |  |
| bgeL_logreg@match_frac | 0.862 | $0.00451 | $0.00000 | 0.00 / 0.00 | 0.87 [0.83, 0.91] | 0.846 |  |
| bge_knn@budget_mid | 0.749 | $0.00198 | $0.00000 | 0.00 / 0.00 | 0.88 [0.85, 0.92] | 0.798 |  |
| bge_knn@match_frac | 0.891 | $0.00820 | $0.00000 | 0.00 / 0.00 | 0.88 [0.85, 0.92] | 0.896 |  |
| bge_logreg@budget_mid | 0.743 | $0.00118 | $0.00000 | 0.00 / 0.00 | 0.89 [0.85, 0.92] | 0.735 |  |
| bge_logreg@match_frac | 0.834 | $0.00504 | $0.00000 | 0.00 / 0.00 | 0.89 [0.85, 0.92] | 0.854 |  |
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
| jev-4k_choice>judge-sonnet@0.3 | 0.883 | $0.00804 | $0.00111 | 0.34 / 3.14 |  |  | deferred 21% |
| jev-4k_choice>judge-sonnet@0.5 | 0.898 | $0.00938 | $0.00211 | 0.39 / 3.21 |  |  | deferred 41% |
| jev-4k_choice>judge-sonnet@0.7 | 0.885 | $0.01050 | $0.00345 | 3.08 / 3.24 |  |  | deferred 68% |
| jev-4k_choice>judge-sonnet@0.9 | 0.874 | $0.01154 | $0.00457 | 3.10 / 3.26 |  |  | deferred 90% |
| jev-4k_choice>judge-luna@0.3 | 0.821 | $0.00432 | $0.00013 | 0.34 / 3.04 |  |  | deferred 21% |
| jev-4k_choice>judge-luna@0.5 | 0.813 | $0.00434 | $0.00018 | 0.39 / 3.11 |  |  | deferred 41% |
| jev-4k_choice>judge-luna@0.7 | 0.791 | $0.00373 | $0.00023 | 2.98 / 3.14 |  |  | deferred 68% |
| jev-4k_choice>judge-luna@0.9 | 0.777 | $0.00326 | $0.00028 | 3.00 / 3.16 |  |  | deferred 90% |

Paired bootstrap vs jev-4k_choice@match_frac (mean diff, 95% CI):

- judge-sonnet: quality +0.004 [-0.021, +0.030], cost +0.00531 [+0.00406, +0.00658]
- judge-luna: quality -0.096 [-0.121, -0.070], cost -0.00346 [-0.00465, -0.00229]

Fallback vs jev-4k_choice alone at the same cost: quality minus jev-4k_choice's cost-quality
hull at the fallback's cost, with Jev's threshold chosen on the test set itself
(optimistic for Jev, so conservative for the fallback). Bootstrap over examples.

- jev-4k_choice>judge-sonnet@0.3: -0.011 [-0.025, +0.001]
- jev-4k_choice>judge-sonnet@0.5: -0.013 [-0.030, +0.002]
- jev-4k_choice>judge-sonnet@0.7: -0.036 [-0.058, -0.019]
- jev-4k_choice>judge-sonnet@0.9: -0.057 [-0.078, -0.036]
- jev-4k_choice>judge-luna@0.3: -0.023 [-0.040, -0.012]
- jev-4k_choice>judge-luna@0.5: -0.031 [-0.048, -0.017]
- jev-4k_choice>judge-luna@0.7: -0.045 [-0.065, -0.026]
- jev-4k_choice>judge-luna@0.9: -0.053 [-0.078, -0.032]

Tier metrics (RowPass / TrajPass):

- judge-sonnet: 0.874 / 0.691
- judge-luna: 0.774 / 0.613
- jev-4k_choice@match_frac: 0.870 / 0.657
