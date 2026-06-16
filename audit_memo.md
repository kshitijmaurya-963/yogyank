### Issues in the broken training script

1. Data Leakage
- Business policy (pm_kisan -150) is applied before the split. The target was modified on the full dataset before train/test split. The test set now has policy-adjusted targets, so R2 measures model + policy as a unit. It will not be possible to audit them sperately in this case.
- defaulted_in_next_12_months is used as a feature. This is a future outcome and it might not exist at scoring time. 

2. Improper Data Splitting
- Random shuffle split on temporal data spaning from 2022 to 2024. Future rows can end up in training, past rows in test. This inflates validation metrics and doesn't simulate real deployment (score a 2025 farmer using a 2022-trained model).

3. No artifact handling
- Single LabelEncoder reused across two columns, encoder.fit_transform(X["crop_type"]) then encoder.fit_transform(X["pm_kisan_status"]). Each .fit_transform resets the encoder. The crop_type encoding is thrown away; it's encoded with pm_kisan's vocabulary. This creates corrupted artifacts.
- Only the model artifacts are saved, not the preprocessing artifacts. The encoder state is never saved. At inference time you can't reproduce the same integer mappings. New categories will silently get wrong encodings.

4. Preprocessing
- Missing values in rainfall_deviation_pct and ndvi_score are not handled. Training can fail or behave unpredictably depending on the model, library versions or change in appereance of null values. 


### How the fixed baseline addresses these issues
- defaulted_in_next_12_months was removed from the feature list. Only variables assumed to be available at scoring time are used.
- The PM-KISAN policy was moved outside the model and implemented in apply_business_policy(). The model output and business adjustment are now separated and can be audited independently.
- The random train/test split was replaced with a temporal split. Training uses 2022 and 2023 data, while evaluation is performed on 2024 only. This better simulates future scoring.
- LabelEncoder was replaced with OrdinalEncoder inside a ColumnTransformer. Each categorical feature is encoded independently and unknown categories are handled safely.
- SimpleImputer(strategy="median") was added to handle missing values. Imputation is fit on training data only and applied through the pipeline.
- The full sklearn Pipeline (preprocessing + model) is saved using joblib.dump(). This ensures training and inference use the same transformations.

### Remaining Limitations
- Validation is based on a single holdout year (2024). The reported metrics may vary if additional years become available.
- Feature computation windows are not provided. Features such as rainfall_deviation_pct and ndvi_score are assumed to be available before the application date. Residual leakage cannot be completely ruled out without feature lineage.
- Hyperparameter tuning was intentionally not performed. The focus of this baseline is fixing engineering issues and improving validation rather than maximizing predictive performance.
