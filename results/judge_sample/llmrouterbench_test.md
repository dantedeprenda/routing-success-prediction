# Judge sample: llmrouterbench_test

1328 stratified test examples (seed 0; ids in `llmrouterbench_test_ids.txt`). Decision routers use the thresholds they picked on labeled train data (raw variant); judges and cascades need none. Cost = model + router.
Judges saw model ids: deepseek-v3.1-terminus, deepseek-r1-0528, gpt-5.

| router | quality | cost | router $/call | latency p50 / p95 s | AUROC cheap-ok | Jev-curve quality at this cost | notes |
|---|---|---|---|---|---|---|---|
| always:deepseek-v3.1-terminus | 0.742 | $0.00112 | $0.00000 | 0.00 / 0.00 |  | 0.000 |  |
| always:deepseek-r1-0528 | 0.785 | $0.01449 | $0.00000 | 0.00 / 0.00 |  | 0.795 |  |
| always:gpt-5 | 0.807 | $0.02508 | $0.00000 | 0.00 / 0.00 |  | 0.807 |  |
| judge-sonnet | 0.790 | $0.02324 | $0.00452 | 2.34 / 7.55 |  | 0.806 | unparsed 1, errors 0, timeouts >9s 53, new spend $0.00 |
| judge-luna | 0.794 | $0.01551 | $0.00012 | 2.51 / 4.18 |  | 0.796 | unparsed 0, errors 0, timeouts >9s 6, new spend $0.00 |
| bgeL_knn@budget_mid | 0.765 | $0.01140 | $0.00000 | 0.00 / 0.00 | 0.74 [0.71, 0.77] | 0.790 |  |
| bgeL_knn@match_frac | 0.783 | $0.01537 | $0.00000 | 0.00 / 0.00 | 0.74 [0.71, 0.77] | 0.796 |  |
| bgeL_logreg@budget_mid | 0.769 | $0.01224 | $0.00000 | 0.00 / 0.00 | 0.75 [0.72, 0.78] | 0.791 |  |
| bgeL_logreg@match_frac | 0.781 | $0.01600 | $0.00000 | 0.00 / 0.00 | 0.75 [0.72, 0.78] | 0.797 |  |
| bge_knn@budget_mid | 0.760 | $0.01111 | $0.00000 | 0.00 / 0.00 | 0.74 [0.70, 0.77] | 0.789 |  |
| bge_knn@match_frac | 0.775 | $0.01459 | $0.00000 | 0.00 / 0.00 | 0.74 [0.70, 0.77] | 0.795 |  |
| bge_logreg@budget_mid | 0.772 | $0.01279 | $0.00000 | 0.00 / 0.00 | 0.75 [0.72, 0.78] | 0.792 |  |
| bge_logreg@match_frac | 0.783 | $0.01620 | $0.00000 | 0.00 / 0.00 | 0.75 [0.72, 0.78] | 0.797 |  |
| category_best@budget_mid | 0.778 | $0.00864 | $0.00000 | 0.00 / 0.00 | 0.69 [0.66, 0.71] | 0.779 |  |
| category_best@match_frac | 0.807 | $0.02508 | $0.00000 | 0.00 / 0.00 | 0.69 [0.66, 0.71] | 0.807 |  |
| jev-2k_choice@budget_mid | 0.787 | $0.01337 | $0.00005 | 0.29 / 0.44 | 0.74 [0.71, 0.77] | 0.793 |  |
| jev-2k_choice@match_frac | 0.790 | $0.01421 | $0.00005 | 0.29 / 0.44 | 0.74 [0.71, 0.77] | 0.794 |  |
| jev-2k_difficulty@budget_mid | 0.791 | $0.01419 | $0.00005 | 0.29 / 0.44 | 0.67 [0.65, 0.70] | 0.794 |  |
| jev-2k_difficulty@match_frac | 0.794 | $0.01528 | $0.00005 | 0.29 / 0.44 | 0.67 [0.65, 0.70] | 0.796 |  |
| jev-2k_features@budget_mid | 0.771 | $0.01382 | $0.00005 | 0.29 / 0.44 | 0.82 [0.79, 0.85] | 0.794 |  |
| jev-2k_features@match_frac | 0.791 | $0.01734 | $0.00005 | 0.29 / 0.44 | 0.82 [0.79, 0.85] | 0.799 |  |
| jev-2k_tier_nouls@budget_mid | 0.785 | $0.01389 | $0.00005 | 0.29 / 0.44 | 0.75 [0.72, 0.78] | 0.794 |  |
| jev-2k_tier_nouls@match_frac | 0.791 | $0.01648 | $0.00005 | 0.29 / 0.44 | 0.75 [0.72, 0.78] | 0.798 |  |
| jev_choice@budget_mid | 0.788 | $0.01321 | $0.00005 | 0.29 / 0.44 | 0.74 [0.71, 0.77] |  |  |
| jev_choice@match_frac | 0.790 | $0.01420 | $0.00005 | 0.29 / 0.44 | 0.74 [0.71, 0.77] |  |  |
| jev_difficulty@budget_mid | 0.793 | $0.01437 | $0.00005 | 0.29 / 0.44 | 0.67 [0.65, 0.70] | 0.794 |  |
| jev_difficulty@match_frac | 0.794 | $0.01564 | $0.00005 | 0.29 / 0.44 | 0.67 [0.65, 0.70] | 0.796 |  |
| jev_features@budget_mid | 0.772 | $0.01404 | $0.00005 | 0.29 / 0.44 | 0.82 [0.80, 0.85] | 0.794 |  |
| jev_features@match_frac | 0.794 | $0.01758 | $0.00005 | 0.29 / 0.44 | 0.82 [0.80, 0.85] | 0.800 |  |
| jev_tier_nouls@budget_mid | 0.785 | $0.01384 | $0.00005 | 0.29 / 0.44 | 0.75 [0.72, 0.78] | 0.794 |  |
| jev_tier_nouls@match_frac | 0.790 | $0.01644 | $0.00005 | 0.29 / 0.44 | 0.75 [0.72, 0.78] | 0.798 |  |
| knn10@budget_mid | 0.757 | $0.01027 | $0.00000 | 0.00 / 0.00 | 0.72 [0.68, 0.75] | 0.786 |  |
| knn10@match_frac | 0.776 | $0.01422 | $0.00000 | 0.00 / 0.00 | 0.72 [0.68, 0.75] | 0.794 |  |
| laya_choice@budget_mid | 0.783 | $0.01405 | $0.00000 | 1.50 / 4.30 | 0.60 [0.57, 0.64] | 0.794 |  |
| laya_choice@match_frac | 0.794 | $0.01869 | $0.00000 | 1.50 / 4.30 | 0.60 [0.57, 0.64] | 0.801 |  |
| laya_difficulty@budget_mid | 0.783 | $0.01428 | $0.00000 | 1.50 / 4.30 | 0.37 [0.34, 0.40] | 0.794 |  |
| laya_difficulty@match_frac | 0.790 | $0.01795 | $0.00000 | 1.50 / 4.30 | 0.37 [0.34, 0.40] | 0.800 |  |
| laya_features@budget_mid | 0.777 | $0.01427 | $0.00000 | 1.50 / 4.30 | 0.69 [0.66, 0.73] | 0.794 |  |
| laya_features@match_frac | 0.795 | $0.01978 | $0.00000 | 1.50 / 4.30 | 0.69 [0.66, 0.73] | 0.803 |  |
| laya_tier_nouls@budget_mid | 0.777 | $0.01241 | $0.00000 | 1.50 / 4.30 | 0.38 [0.35, 0.41] | 0.791 |  |
| laya_tier_nouls@match_frac | 0.785 | $0.01626 | $0.00000 | 1.50 / 4.30 | 0.38 [0.35, 0.41] | 0.798 |  |
| tfidf_logreg@budget_mid | 0.762 | $0.01132 | $0.00000 | 0.00 / 0.00 | 0.76 [0.73, 0.80] | 0.789 |  |
| tfidf_logreg@match_frac | 0.748 | $0.00849 | $0.00000 | 0.00 / 0.00 | 0.76 [0.73, 0.80] | 0.778 |  |
| jev_choice>judge-sonnet@0.3 | 0.788 | $0.01608 | $0.00055 | 0.29 / 2.63 |  |  | deferred 11% |
| jev_choice>judge-sonnet@0.5 | 0.793 | $0.01951 | $0.00157 | 0.32 / 2.69 |  |  | deferred 34% |
| jev_choice>judge-sonnet@0.7 | 0.791 | $0.02017 | $0.00231 | 1.81 / 2.73 |  |  | deferred 50% |
| jev_choice>judge-sonnet@0.9 | 0.790 | $0.02186 | $0.00333 | 2.62 / 2.76 |  |  | deferred 72% |
| jev_choice>judge-luna@0.3 | 0.786 | $0.01477 | $0.00007 | 0.29 / 2.81 |  |  | deferred 11% |
| jev_choice>judge-luna@0.5 | 0.791 | $0.01543 | $0.00009 | 0.32 / 2.86 |  |  | deferred 34% |
| jev_choice>judge-luna@0.7 | 0.794 | $0.01500 | $0.00011 | 1.90 / 2.90 |  |  | deferred 50% |
| jev_choice>judge-luna@0.9 | 0.794 | $0.01549 | $0.00014 | 2.79 / 2.94 |  |  | deferred 72% |

Paired bootstrap vs jev_choice@match_frac (mean diff, 95% CI):

- judge-sonnet: quality +0.000 [-0.012, +0.012], cost +0.00904 [+0.00775, +0.01053]
- judge-luna: quality +0.004 [-0.006, +0.014], cost +0.00131 [+0.00052, +0.00219]

Fallback vs jev_choice alone at the same cost: quality minus jev_choice's cost-quality
hull at the fallback's cost, with Jev's threshold chosen on the test set itself
(optimistic for Jev, so conservative for the fallback). Bootstrap over examples.

- jev_choice>judge-sonnet@0.3: -0.009 [-0.018, -0.003]
- jev_choice>judge-sonnet@0.5: -0.009 [-0.022, -0.002]
- jev_choice>judge-sonnet@0.7: -0.012 [-0.027, -0.004]
- jev_choice>judge-sonnet@0.9: -0.015 [-0.027, -0.006]
- jev_choice>judge-luna@0.3: -0.009 [-0.017, -0.004]
- jev_choice>judge-luna@0.5: -0.005 [-0.014, -0.000]
- jev_choice>judge-luna@0.7: -0.001 [-0.011, +0.004]
- jev_choice>judge-luna@0.9: -0.002 [-0.013, +0.003]
