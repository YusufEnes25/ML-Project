# Machine Learning Classification Pipeline

This repository contains a complete, end-to-end machine learning pipeline built for a university assignment. The goal was to process raw tabular data, handle severe class imbalances, and train a highly optimized classification model.

### Key Technical Highlights:
* **Algorithm:** XGBoost Classification (`model.ubj`)
* **Feature Engineering:** Polynomial Features (Degree 2) applied to capture non-linear relationships.
* **Preprocessing:** Strict train/test isolation to prevent data leakage, utilizing `StandardScaler` / `QuantileTransformer`.
* **Hyperparameter Tuning:** 6-fold cross-validation using `GridSearchCV` testing over 300 configurations.
* **Anti-Overfitting:** Implemented L2 Regularization (`reg_lambda`), row/column subsampling, and controlled learning rates.
