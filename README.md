# 📉 Customer Churn Prediction & Retention Optimization

A business-focused machine learning project that predicts telecom customer churn and optimizes retention strategy using probability calibration and threshold-based ROI maximization.

**ROC-AUC: 0.84 | Optimal Threshold: ~42% | Max Expected Value: $18,830 on held-out test set**

---

## 🧠 The Business Problem

A telecom company loses significant revenue every time a customer cancels. The challenge isn't just predicting *who* will churn — it's deciding *who to target with a retention offer* in a way that's actually profitable.

Targeting everyone is wasteful. Targeting only the highest-risk customers misses the recoverable middle. This project builds a model that answers: **"Given this customer's churn probability, does it make financial sense to send them a retention offer?"**

**Business assumptions:**
- Customer Lifetime Value: **$800**
- Retention offer cost: **$100 per customer**
- Offer success rate: **30%** (customers who accept and stay)
- Goal: maximize Expected Value = `(churn_prob × save_rate × LTV) − offer_cost`

---

## 📊 Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 features

Features include customer demographics, account information (contract type, tenure, payment method), services subscribed (phone, internet, add-ons), and monthly/total charges.

**Target variable:** `Churn` (Yes/No) — 26.5% churn rate (class imbalance addressed with `class_weight='balanced'`)

---

## 🔍 Key EDA Findings

| Driver | Finding |
|---|---|
| Contract Type | Month-to-month customers churn at **42%** vs 3% for two-year contracts |
| Tenure | **First 12 months** have >50% churn — drops sharply after year one |
| Internet Service | Fiber optic customers churn ~30-35%, DSL ~20% |
| Service Bundles | Customers without OnlineSecurity or TechSupport churn significantly more |
| Payment Method | Electronic check users churn **3x more** than autopay users |
| Monthly Charges | Higher charges ($70–110 range) correlate with higher churn risk |

---

## ⚙️ Modeling Pipeline

```
Raw Data
   │
   ├─ Numeric features → Median Imputer → ColumnTransformer
   ├─ Categorical features → Mode Imputer → OneHotEncoder
   │
   └─ Logistic Regression (class_weight='balanced')
            │
            └─ CalibratedClassifierCV (isotonic, cv=5)
                        │
                        └─ Threshold Optimizer → Scored Output File
```

**Why Logistic Regression?**
Interpretable, probability-native, and a strong baseline for a business decision problem. The goal is reliable probability estimates — not squeezing out marginal accuracy from a black-box model.

**Why calibrate?**
Raw model probabilities aren't reliable for threshold-based decisions. Isotonic calibration reduced the Brier Score from 0.169 → 0.139, ensuring "40% predicted churn" actually corresponds to ~40% observed churn.

---

## 📈 Results

| Metric | Uncalibrated | Calibrated |
|---|---|---|
| ROC-AUC | 0.842 | 0.841 |
| PR-AUC | 0.633 | 0.628 |
| Brier Score | 0.169 | **0.139** ✅ |

**Threshold Optimization:**
- Optimal threshold: **~42% churn probability**
- Customers targeted at this threshold: **401**
- Max Expected Value on test set: **$18,830**

Segment-level performance:

| Contract Type | N | Churn Rate | ROC-AUC |
|---|---|---|---|
| Month-to-month | 773 | 42.6% | 0.741 |
| One year | 300 | 12.0% | 0.746 |
| Two year | 336 | 2.7% | 0.765 |

---

## 💡 Business Recommendations

1. **Contract migration is the highest-ROI lever** — converting month-to-month customers to annual contracts is worth prioritizing over blanket discounts
2. **Onboarding investment in year one** — churn risk is highest in the first 12 months; front-loading engagement here pays off
3. **Bundle cross-selling** — customers without OnlineSecurity and TechSupport churn far more; targeted bundle offers serve both retention and revenue goals
4. **Payment method nudge** — electronic check users churn 3x more; nudging toward auto-pay is low-cost and high-signal

---

## 📁 Repository Structure

```
├── CustomerChurn.ipynb       # Full end-to-end analysis and modeling notebook
├── Telco-Customer-Churn.csv  # Source dataset
└── README.md
```

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` · `seaborn`

Key techniques: Pipeline + ColumnTransformer, CalibratedClassifierCV (isotonic), permutation importance, threshold optimization via expected value maximization, segment-level model evaluation

---

## 🚀 How to Run

```bash
git clone https://github.com/paramdeepnijjer-blip/Customer-Churn-ML
cd Customer-Churn-ML
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook CustomerChurn.ipynb
```

---

## 👤 Author

**Paramdeep Nijjer** — M.S. Data Science, Boston University  
[LinkedIn](https://www.linkedin.com/in/paramdeepnijjer/) · [GitHub](https://github.com/paramdeepnijjer-blip)
