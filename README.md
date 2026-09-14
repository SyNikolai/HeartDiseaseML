# HeartDiseaseML

Predicting the presence of heart disease from clinical measurements, using feature selection and an optimized Support Vector Machine.

## Overview

This project asks a simple question: **can we confidently predict heart disease from routine clinical test results?**

Using a dataset of 270 patients who underwent cardiac catheterization, the project explores which clinical variables are most predictive, then builds and tunes a classifier to make that prediction — reaching **88.89% accuracy** on held-out test data with a Bayesian-optimized SVM.

## Dataset

The [Heart Disease Prediction dataset](https://www.kaggle.com/datasets/thedevastator/predicting-heart-disease-risk-using-clinical-var) (originally from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)) contains 270 patients, each described by 13 clinical variables and a binary target (`Heart Disease`: Presence / Absence).

| Feature | Description |
|---|---|
| Age | Age in years |
| Sex | 1 = male, 0 = female |
| Chest pain type | 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic |
| BP | Resting blood pressure |
| Cholesterol | Serum cholesterol (mg/dl) |
| FBS over 120 | Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no) |
| EKG results | 0 = normal, 1 = ST-T wave abnormality, 2 = probable/definite left ventricular hypertrophy |
| Max HR | Maximum heart rate achieved during exercise |
| Exercise angina | Exercise-induced angina (1 = yes, 0 = no) |
| ST depression | ST depression induced by exercise relative to rest |
| Slope of ST | Slope of the peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping) |
| Number of vessels fluro | Number of major vessels (1–3) colored by fluoroscopy |
| Thallium | Thallium stress test result (3 = normal, 6 = fixed defect, 7 = reversible defect) |
| Heart Disease | Target: Presence / Absence |

## Approach

**1. Exploratory analysis**
- Checked for missing values and duplicates (none found)
- Visualized feature distributions and outliers (boxplots, histograms)
- Examined pairwise relationships with a pairplot (`pairplot.png`)
- Computed a Spearman correlation matrix to inspect (non-linear) monotonic relationships between features

**2. Feature selection**
- Trained a `RandomForestClassifier` on the full feature set and used `SelectFromModel` to identify the subset of features that contribute most to prediction, since the pairplot/correlation matrix didn't reveal strong linear relationships that would justify a simpler approach

**3. Predictive model**
- Selected a **Support Vector Machine (SVC)** for the final classifier, chosen for its ability to model non-linear decision boundaries via kernel methods and its resistance to overfitting on a relatively small dataset
- Tuned `C`, `gamma`, `degree`, and `coef0` using **Bayesian optimization** (`skopt.BayesSearchCV`) instead of a naive grid search, for faster convergence to good hyperparameters
- Evaluated the final model with accuracy, F1 score, and a confusion matrix on a stratified 70/30 train-test split

## Results

The Bayesian-optimized SVM achieved **88.89% accuracy** on the test set, using only the subset of features identified as most important by the random forest.

Notably, `Sex` was among the features *not* selected as important — an interesting result worth further investigation given known clinical associations between sex and cardiovascular risk.

## Future Work

- Compare against ensemble methods such as gradient boosting (e.g. XGBoost, LightGBM)
- Cross-validate feature selection itself, rather than selecting once on a single train/test split
- Test on a larger, more diverse patient population to assess generalizability

## Repository Contents

| File | Description |
|---|---|
| `heartdisease.ipynb` | Full analysis notebook: EDA, feature selection, model training, and evaluation |
| `Heart_Disease_Prediction.csv` | Raw dataset |
| `pairplot.png` | Pairwise feature plot, colored by diagnosis |
| `LICENSE` | MIT License |

## Getting Started

```bash
pip install pandas numpy scikit-learn scikit-optimize matplotlib seaborn
jupyter notebook heartdisease.ipynb
```

## License

This project is licensed under the [MIT License](LICENSE).

## Disclaimer

This project is for educational and research purposes only. It is not a diagnostic tool and should not be used to inform real medical decisions.
