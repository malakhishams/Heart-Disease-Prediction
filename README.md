# Heart Disease Prediction

A machine learning project submitted as a final project for our Artificial Intelligence course at the faculty of computer and information sciences - Ain Shams University. The goal is to predict the presence of heart disease from clinical and demographic patient data. Includes full EDA, preprocessing, model comparison, and an interactive web interface built with NiceGUI.

---

## Project Structure

```
heart-disease-prediction/
│
├── data/
│   ├── train_data.csv
│   └── test_data.csv
│
├── notebooks/
│   ├── heart_disease.ipynb   # EDA, preprocessing & model comparison
│   └── model_pipeline.ipynb                                  # Final sklearn pipeline & serialization
|
├── models/
│   └── model.pkl
|
├── app/
│   └── app.py                # NiceGUI web interface
│
├── reports/
│   └── project_report.pdf     # Full project report
│
├── requirements.txt
└── README.md
```

---

## Dataset

- **224 training samples**, **16 features**
- Features include clinical measurements (BP, cholesterol, max HR, ST depression), cardiac test results (EKG, thallium, fluoroscopy vessels), and demographic/lifestyle factors (age, gender, work type, smoking status)
- **Target:** `Heart Disease` — Yes / No (balanced, no class imbalance)

The dataset is at Kaggle: nadaahmedsamir/heart-disease-uni-dataset

---

## Methodology

### EDA & Key Findings
- **Thallium scan results** and **chest pain type** had the strongest association with heart disease (Cramér's V)
- **Number of fluoroscopy vessels** and **max heart rate** were the most correlated numerical features (Pearson)
- **Gender** was a strong predictor: most females in the dataset had heart disease, most males did not
- No class imbalance was found, so standard accuracy is a reliable metric

### Preprocessing Pipeline
| Step | Details |
|---|---|
| Drop ID column | Removed before training |
| Missing values | Age → median; Gender, work_type → mode; smoking_status → `"Unknown"` |
| Duplicates | 9 dropped from train, 1 from test |
| Rare categories | Rows with `children` / `Never_worked` in `work_type` dropped |
| Outlier clipping | IQR-based clipping on `BP` and `Cholesterol` |
| Encoding | Label encoding for `Gender`; One-Hot for `work_type`, `smoking_status`, `Thallium` |
| Scaling | MinMax scaling on all numerical features |

All preprocessing is fit on train data only to prevent data leakage.

### Model Comparison

| Model | Train Acc. | Test Acc. | Gap |
|---|---|---|---|
| Decision Tree | 86.10% | 84.91% | 1.19% |
| Random Forest | 90.00% | 86.00% | 4.00% |
| SVM (linear) | 85.00% | 84.00% | 1.00% |
| KNN (k=13) | 85.00% | 83.00% | 2.00% |
| Logistic Regression | 85.20% | 84.90% | 0.30% |
| **XGBoost** ✅ | **92.38%** | **88.68%** | **3.70%** |

### Final Model — XGBoost
- **Test accuracy:** 88.7% | **Macro F1:** 0.88
- **Hyperparameters:** `n_estimators=200`, `max_depth=3`, `learning_rate=0.01`, `colsample_bytree=1.0`, `subsample=0.8`
- Decision threshold lowered from 0.50 → 0.40 to improve recall (0.77 → 0.82) at a minor precision cost

---

## Installation

```bash
git clone https://github.com/malakhishams/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
```

---

## Running the Web App

Make sure `model.pkl` is in the same directory as `app.py` (or update the path inside the file), then:

```bash
cd app
python app.py
```

Open your browser at `http://localhost:8080`

The interface collects all 15 patient features across four sections — Basic Information, Heart Symptoms, Advanced Tests, and Additional Information — and returns a prediction with the probability of heart disease.

---

## Running the Notebooks

The EDA notebook downloads the dataset via `kagglehub`. Make sure you have your Kaggle API credentials configured before running:

```bash
# Configure Kaggle credentials first
# https://www.kaggle.com/docs/api#authentication

jupyter notebook notebooks/Heart_Disease_last_Version_updated_copy_submission.ipynb
```

The pipeline notebook (`last-model-pipeline.ipynb`) reads local CSV files — update the file paths at the top of the notebook to point to your `data/` directory before running.

---

## Team Contributions

| Task | Team Member |
|---|---|
| Data & Correlation Analysis | [Malak Hisham](https://github.com/malakhishams) |
| Data Preprocessing | [Nada Ahmed](https://github.com/Nada-ASamir) |
| GridSearch & Model Selection | [Rana Ahmed](https://github.com/ranaahmedhussein), [Menna Mahmoud](https://github.com/mennatullah-9), [Doha Ayman](https://github.com/doha-aymann), [Alia Wael](https://github.com/Alia-elnaggar), [Malak Hisham](https://github.com/malakhishams) and [Malak Mahmoud](https://github.com/malak-mahm0ud) |
| Web UI Development | [Malak Mahmoud](https://github.com/malak-mahm0ud) and [Menna Mahmoud](https://github.com/mennatullah-9) |
| Model Pipeline and Integration | [Nada Ahmed](https://github.com/Nada-ASamir) and [Malak Hisham](https://github.com/maakhishams) |

Submitted as a final project for the Artificial Intelligence course.

A big thank you to every team member for their hard work, collaboration, and dedication throughout this project. This would not have been possible without each one of you. 
