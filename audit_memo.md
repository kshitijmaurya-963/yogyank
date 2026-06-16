## Issues in the broken training script

1. Data Leakage
- Business policy (pm_kisan -150) is applied before the split. The target was modified on the full dataset before train/test split. The test set now has policy-adjusted targets, so R2 measures model + policy as a unit. It will not be possible to audit them sperately in this case.
- defaulted_in_next_12_months is used as a feature. This is a future outcome and it might not exist at scoring time. 

2. Improper Data Splitting
- Random shuffle split on temporal data spaning from 2022 to 2024. Future rows can end up in training, past rows in test. This inflates validation metrics and doesn't simulate real deployment (score a 2025 farmer using a 2022-trained model).

3. No artifact handling
- Single LabelEncoder reused across two columns, encoder.fit_transform(X["crop_type"]) then encoder.fit_transform(X["pm_kisan_status"]). Each .fit_transform resets the encoder. The crop_type encoding is thrown away; it's encoded with pm_kisan's vocabulary. This creates corrupted artifacts.
- Only the model artifacts are saved, not the preprocessing artifacts. The encoder state is never saved. At inference time you can't reproduce the same integer mappings. New categories will silently get wrong encodings.
