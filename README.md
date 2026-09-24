# 🚢 Titanic Survival Prediction — End-to-End ML Project

An end-to-end machine-learning project that predicts whether a passenger survived the Titanic disaster, from problem understanding to a deployed **Streamlit** web app.

| | |
|---|---|
| **Problem type** | Supervised learning — binary classification |
| **Dataset** | [Kaggle Titanic](https://www.kaggle.com/c/titanic) `train.csv` (891 passengers, 12 columns) |
| **Best model** | Random Forest (tuned with GridSearchCV) inside a scikit-learn `Pipeline` |
| **Test accuracy** | **~82.7%** (ROC-AUC ~0.86) |
| **Demo** | Streamlit app (`app.py`) |

## 📌 Project Workflow

1. **Problem understanding** — predict `Survived` (0/1) from passenger details.
2. **EDA** — survival by sex, class, age, fare, family size.
3. **Preprocessing** — median/most-frequent imputation, scaling, one-hot encoding using `Pipeline` + `ColumnTransformer` (no data leakage).
4. **Feature engineering** — `Title` (from name), `FamilySize`, `IsAlone`.
5. **Modeling** — Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes compared with 5-fold stratified cross-validation.
6. **Tuning** — `GridSearchCV` on Logistic Regression and Random Forest; winner chosen by CV score.
7. **Evaluation** — accuracy, precision, recall, F1, confusion matrix, ROC-AUC, permutation importance.
8. **Deployment** — trained pipeline saved with `joblib` and served through Streamlit.

## 📊 Key Results

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression | 82.3% | 83.8% |
| KNN | 80.3% | 81.6% |
| Decision Tree | 80.3% | 79.9% |
| Naive Bayes | 79.2% | 78.8% |
| Random Forest (default) | 79.2% | 79.9% |
| **Random Forest (tuned)** | **82.9%** | **82.7%** |

The tuned model is selected using **cross-validation**, not the test set, so the 82.7% test score is an honest estimate.

**Main insights**
- Women survived at **74%** vs **19%** for men.
- Survival by class: **63%** (1st) → **47%** (2nd) → **24%** (3rd).
- Children under 12 had the highest survival rate (~58%).
- Adding engineered features raised test accuracy from ~80% (raw features) to ~83%.

## 🗂️ Repository Structure

```
titanic-capstone/
├── app.py                          # Streamlit web app
├── data/train.csv                  # Dataset
├── models/
│   ├── titanic_model.joblib        # Trained pipeline
│   └── metrics.json                # Saved scores & best params
├── notebooks/Titanic_Capstone.ipynb  # Full analysis: code + explanations
├── reports/Titanic_Capstone_Presentation.pptx  # 5–7 minute presentation
├── requirements.txt
└── README.md
```

## ▶️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/titanic-capstone.git
cd titanic-capstone

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Re-run the analysis (optional — creates models/titanic_model.joblib)
jupyter notebook notebooks/Titanic_Capstone.ipynb

# 5. Launch the web app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Enter a passenger's details and click **Predict survival**.

## ⚠️ Limitations & Future Work

- Small dataset (891 rows) → accuracy estimates are uncertain by roughly ±3%.
- Survivor recall (72%) is lower than non-survivor recall (89%); threshold tuning could help.
- Future: engineer deck/ticket-group features, try XGBoost/LightGBM, and submit to the Kaggle leaderboard.

## 🛠️ Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Streamlit · joblib

## 📄 License & Credits

Dataset: Kaggle *Titanic — Machine Learning from Disaster*. Project built as an ML capstone.
# Titanic_Survival_Predictor
