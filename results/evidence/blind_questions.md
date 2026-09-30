# X2 routing arm: label-blind questions (pre-registered, preregistration/v2.md)

Failed Jev calls (imputed 0.5): {1: 0, 2: 0, 3: 0}. Bootstrap: 2000 replicates over question sets, label seeds and test prompts (stratified by category).

**Primary endpoint** (k = 25, mean over Q-blind-1..3 of Q-blind - bge_knn): **+0.130 [+0.080, +0.177]**; success criterion (lower bound > 0): **MET**.

| k | Q-author (v1) | Q-blind 1 / 2 / 3 | bge_knn | Q-blind - bge_knn | Q-blind - Q-author |
|---|---|---|---|---|---|
| 10 | 0.677 | 0.654 / 0.660 / 0.638 | 0.500 | +0.151 [+0.100, +0.194] | -0.026 [-0.066, +0.025] |
| 25 | 0.729 | 0.659 / 0.663 / 0.692 | 0.542 | +0.130 [+0.080, +0.177] | -0.058 [-0.107, -0.024] |
| 50 | 0.721 | 0.691 / 0.691 / 0.708 | 0.531 | +0.166 [+0.125, +0.208] | -0.025 [-0.055, +0.007] |
| 100 | 0.749 | 0.708 / 0.715 / 0.723 | 0.562 | +0.153 [+0.118, +0.190] | -0.034 [-0.058, -0.012] |
| 200 | 0.744 | 0.707 / 0.702 / 0.708 | 0.559 | +0.146 [+0.103, +0.191] | -0.038 [-0.058, -0.018] |
| all | 0.757 | 0.724 / 0.724 / 0.727 | 0.651 | +0.074 [+0.030, +0.120] | -0.031 [-0.052, -0.012] |

9 unseen targets, k = 25, Q-blind - bge_knn (mean over targets and sets): +0.128 [+0.095, +0.164].

Closest v1 question per label-blind question (bge-large cosine):

| set | question | closest v1 | cosine |
|---|---|---|---|
| 1 | is_multiple_choice | ambiguous | 0.74 |
| 1 | requires_multi_step_reasoning | multistep | 0.79 |
| 1 | has_explicit_final_answer_format | strict_format | 0.75 |
| 1 | includes_code_snippet | code | 0.70 |
| 1 | requires_code_execution_knowledge | state_tracking | 0.73 |
| 1 | requires_external_or_specialized_knowledge | expert | 0.74 |
| 1 | has_prior_agent_steps | ambiguous | 0.79 |
| 1 | is_long_or_complex_prompt | long_context | 0.75 |
| 1 | requires_precise_numeric_computation | math | 0.76 |
| 1 | is_ambiguous_or_underspecified | ambiguous | 0.79 |
| 1 | requires_tool_or_environment_interaction | routine_tool_use | 0.69 |
| 1 | has_verifiable_ground_truth | ambiguous | 0.74 |
| 1 | requires_long_form_explanation | math | 0.74 |
| 2 | requires_multi_step_reasoning | multistep | 0.76 |
| 2 | has_explicit_final_answer_format | strict_format | 0.76 |
| 2 | involves_code_writing | code | 0.78 |
| 2 | involves_code_debugging | code | 0.75 |
| 2 | provides_prior_agent_steps | agent_action | 0.78 |
| 2 | requires_external_knowledge | expert | 0.76 |
| 2 | is_multiple_choice | ambiguous | 0.74 |
| 2 | has_large_context | code | 0.74 |
| 2 | requires_numeric_precision | math | 0.78 |
| 2 | is_open_ended | ambiguous | 0.80 |
| 2 | requires_tool_or_execution_simulation | code | 0.70 |
| 2 | has_ambiguous_or_underspecified_requirements | ambiguous | 0.76 |
| 2 | requires_proof_or_derivation | math | 0.71 |
| 3 | requires_multi_step_reasoning | math | 0.78 |
| 3 | has_explicit_final_answer_format | strict_format | 0.76 |
| 3 | involves_code_writing | code | 0.80 |
| 3 | includes_existing_code_or_error | code | 0.73 |
| 3 | requires_external_knowledge | expert | 0.73 |
| 3 | is_multiple_choice | ambiguous | 0.75 |
| 3 | has_prior_agent_steps | agent_action | 0.78 |
| 3 | requires_numeric_computation | math | 0.75 |
| 3 | is_ambiguous_or_underspecified | ambiguous | 0.77 |
| 3 | requires_long_output | long_context | 0.73 |
| 3 | requires_domain_specific_terminology | expert | 0.67 |
| 3 | involves_constraints_or_edge_cases | strict_format | 0.71 |
| 3 | is_self_contained | strict_format | 0.73 |
