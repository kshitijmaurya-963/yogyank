## Tools Used

* ChatGPT
* OpenAI Codex

## Where I Used Them

* Brainstorming safer alternatives to the original training pipeline.
* Refactoring the training script into smaller functions (`validate_schema`, `temporal_split`, `evaluate`, `save_artifacts`).
* Reviewing the preprocessing and pipeline persistence approach.

## Prompts Used

1. "Review this training script and identify data leakage, validation and preprocessing issues."
2. "Refactor this script into modular functions while keeping the logic simple."
3. "What artifacts should be saved for a reproducible sklearn + XGBoost baseline?"

## Suggestions Accepted

- Use a temporal split instead of random shuffle.
- Persist the full sklearn Pipeline instead of only the model.
- Move PM-KISAN policy outside the model.
- Add schema validation and save minimal metadata artifacts.

## Suggestions Rejected or Modified

- SHAP-based reason codes were initially considered but omitted due to dependency issues (`shap`/`numba`/`llvmlite`) on the local environment and because explainability was not required for the baseline.
- Hyperparameter tuning was intentionally not performed to keep the comparison focused on bug fixes rather than optimisation.

## Personally Verified

- Leakage reasoning and exclusion of `defaulted_in_next_12_months`.
- Temporal validation strategy.
- Preprocessing and imputation are fit only on training data.
- Pipeline persistence and reloadability.
- Artifact generation (`pipeline.pkl`, `feature_list.json`, `validation_summary.json`).
- End-to-end training and metric output.
