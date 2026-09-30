# Labels needed to match the decision model's 25-label discrimination

## llmrouterbench_test

Target: jev:features at k=25 = 0.729. Labeling one prompt costs $0.0427 with every pool model, $0.00117 with the cheap model only. Train split: 3,606 prompts.

| router | best mean AUROC (any k) | labels to reach target | labeling cost |
|---|---|---|---|
| jev:features | 0.757 | 25 | $1.07 (all models) / $0.03 (cheap only) |
| tfidf_logreg | 0.664 | not reached with all 3,606 | > $154.12 (all models) / $4.22 (cheap only) |
| knn10 | 0.621 | not reached with all 3,606 | > $154.12 (all models) / $4.22 (cheap only) |
| bge_logreg | 0.648 | not reached with all 3,606 | > $154.12 (all models) / $4.22 (cheap only) |
| bge_knn | 0.651 | not reached with all 3,606 | > $154.12 (all models) / $4.22 (cheap only) |
| bgeL_logreg | 0.650 | not reached with all 3,606 | > $154.12 (all models) / $4.22 (cheap only) |
| bgeL_knn | 0.645 | not reached with all 3,606 | > $154.12 (all models) / $4.22 (cheap only) |

## twinrouterbench_test

Target: jev-4k:features at k=25 = 0.654. Labeling one prompt costs $0.0316 with every pool model, $0.00098 with the cheap model only. Train split: 500 prompts.

| router | best mean AUROC (any k) | labels to reach target | labeling cost |
|---|---|---|---|
| jev-4k:features | 0.718 | 25 | $0.79 (all models) / $0.02 (cheap only) |
| tfidf_logreg | 0.722 | all 500 | $15.82 (all models) / $0.49 (cheap only) |
| knn10 | 0.598 | not reached with all 500 | > $15.82 (all models) / $0.49 (cheap only) |
| bge_logreg | 0.656 | all 500 | $15.82 (all models) / $0.49 (cheap only) |
| bge_knn | 0.687 | all 500 | $15.82 (all models) / $0.49 (cheap only) |
| bgeL_logreg | 0.591 | not reached with all 500 | > $15.82 (all models) / $0.49 (cheap only) |
| bgeL_knn | 0.714 | all 500 | $15.82 (all models) / $0.49 (cheap only) |

