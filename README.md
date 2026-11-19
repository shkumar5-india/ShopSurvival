# 🏪 ShopSurvival

**AI-Powered Business Status Analysis**

ShopSurvival is a Streamlit web application that uses a trained XGBoost classifier to estimate the recorded **Open** or **Closed** status of a business from its operational, location and local economic information. It returns a prediction along with class probabilities.

> **Disclaimer:** The app predicts *recorded* business status based on the training dataset. It does not guarantee future business survival. The training data is based on US businesses.

---

## Features

- Clean, modern dashboard UI with sidebar, hero banner and card layout
- Business input form grouped into performance and location/economy sections
- Multi-select business categories (loaded dynamically from the model's feature list)
- Predicted status (Open / Closed) with confidence
- Open and Closed probability cards with visual meters
- Probability bar chart and business summary table
- Friendly error handling when model files are missing or mismatched

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend / App | Streamlit |
| Model | XGBoost (`XGBClassifier`) |
| Preprocessing | scikit-learn pipeline, saved with joblib |
| Data handling | pandas |

---

## Project Structure

```
.
├── app.py
├── requirements.txt
├── README.md
└── shopsurvival_models/
    ├── preprocessor.pkl     # fitted preprocessing pipeline
    ├── xgb_model.json       # trained XGBoost model
    └── features.json        # ordered list of model input features
```

All three files in `shopsurvival_models/` must be exported from the **same** training pipeline. Otherwise predictions will fail or be wrong.

---

## Installation

1. **Clone or download** the project.

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:

   ```
   streamlit>=1.28
   pandas
   joblib
   xgboost
   scikit-learn
   ```

   Use the same scikit-learn and XGBoost versions you trained with to avoid pickle or model-loading issues.

4. **Place the model files** in a folder named `shopsurvival_models` next to `app.py`.

---

## Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Usage

1. Fill in the **Business Information** form.
2. Select one or more **Business Categories**.
3. Click **Analyze Business**.
4. Review the predicted status, probabilities, chart and summary.

### Input Fields

| Field | Description |
|---|---|
| Rating × Reviews | Star rating multiplied by review count |
| Review Count | Total number of reviews |
| Number of Categories | How many categories the business belongs to |
| Years in Business | Age of the business in years |
| Number of Check-ins | Total check-ins recorded |
| Has Check-in? | Whether any check-in exists |
| Latitude / Longitude | Business location |
| Per Capita Personal Income | Local income indicator |
| Poverty Rate | Local poverty rate |
| Median Household Income | Local median household income |
| Unemployment Rate | Local unemployment rate |
| Average Weekly Wages | Local average weekly wages |

Enter economic values in the same units used during training (for example percentages vs fractions).

### Sample Input

| Field | Healthy example | Struggling example |
|---|---|---|
| Rating × Reviews | 945 | 24 |
| Review Count | 210 | 8 |
| Number of Categories | 4 | 1 |
| Years in Business | 9 | 1 |
| Number of Check-ins | 320 | 0 |
| Has Check-in? | Yes | No |
| Latitude | 39.952600 | 36.162700 |
| Longitude | -75.165200 | -86.781600 |
| Per Capita Personal Income | 68000 | 41000 |
| Poverty Rate | 11.5 | 19.8 |
| Median Household Income | 72000 | 48000 |
| Unemployment Rate | 4.2 | 7.1 |
| Average Weekly Wages | 1350 | 980 |

---

## How It Works

1. `features.json` defines the full ordered feature list the model expects.
2. User inputs are mapped into a one-row DataFrame. All features default to `0`, numeric fields are filled in, and selected `cat_*` flags are set to `1`.
3. The saved preprocessor transforms the row.
4. XGBoost returns the predicted class and probabilities (class `1` = Open, class `0` = Closed).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Model files missing" error | Make sure `shopsurvival_models/` contains `preprocessor.pkl`, `xgb_model.json` and `features.json` |
| "Prediction failed" error | Re-export all three files from the same training run and match library versions |
| Chart `color` error | Upgrade Streamlit (`pip install -U streamlit`) or remove the `color` argument from `st.bar_chart` |
| Odd predictions | Check that input units match the training data |

---

## Limitations

- Trained on US business data only
- Predicts recorded status, not future survival
- Accuracy depends on the quality and coverage of the training data

---

## License

Add your preferred license here (for example MIT).

---

*ShopSurvival | AI & Machine Learning Project*
