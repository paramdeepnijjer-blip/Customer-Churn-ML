# Customer Churn Prediction & Retention Optimization

A machine learning project that predicts telecom customer churn and optimizes retention campaign strategy to maximize business ROI.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![Tableau](https://img.shields.io/badge/Tableau-Public-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Project Overview

This project goes beyond traditional classification models by **optimizing for business value** rather than just accuracy. Using probability calibration and threshold optimization, the model identifies which customers to target for retention offers to maximize expected ROI.

### Key Results
- **ROC-AUC: 0.84** - Excellent discrimination between churners and non-churners
- **Optimal Threshold: 46%** - Customers above this probability should receive retention offers
- **Business Impact: $XXX,XXX** estimated annual value from optimized targeting
- **Model Calibration: Well-calibrated** probabilities for reliable business decisions

## 📊 Business Problem

Customer acquisition costs 5-25x more than retention. This project addresses:
1. **Who will churn?** Predict customer churn probability
2. **Who should we target?** Optimize retention offer strategy for maximum ROI
3. **Why do they churn?** Identify key drivers for strategic interventions

### Retention Economics
- **Customer Lifetime Value (LTV):** $800
- **Retention Offer Cost:** $100 per customer
- **Offer Success Rate:** 30% (customers who accept and stay)
- **Goal:** Maximize Expected Value = (Probability × 0.30 × $800) - $100

## 🗂️ Repository Structure

```
customer-churn-ml/
├── CustomerChurn.ipynb              # Main analysis notebook
├── scored_customers_calibrated.csv   # Model predictions & risk scores
├── customer_churn_documentation.md   # Detailed project writeup
├── Churn_ML_Dashboard.twb           # Tableau dashboard file
├── data/
│   └── Telco-Customer-Churn.csv     # Original dataset
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.10+
Jupyter Notebook
Tableau Desktop (for dashboard)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR-USERNAME/customer-churn-ml.git
cd customer-churn-ml
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the notebook**
```bash
jupyter notebook CustomerChurn.ipynb
```

### Required Libraries
```python
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## 📈 Methodology

### 1. Data Preparation
- **Dataset:** 7,043 telecom customers with 21 features
- **Target:** Churn (Yes/No)
- **Features:** Demographics, account info, services, billing
- **Cleaning:** Fixed missing TotalCharges, encoded target variable

### 2. Exploratory Data Analysis

**Key Insights:**
- **Contract Type:** Month-to-month customers churn at 14x the rate of two-year contracts (42% vs 3%)
- **Tenure:** First 12 months are highest risk (>50% churn rate)
- **Internet Service:** Fiber optic customers churn more than DSL customers
- **Service Bundles:** Customers without OnlineSecurity, TechSupport, DeviceProtection churn significantly more
- **Payment Method:** Electronic check users churn 3x more than autopay users

### 3. Model Development

**Algorithm:** Logistic Regression with class balancing
- Simple, interpretable, produces probabilities
- `class_weight='balanced'` handles 26.5% churn rate
- L2 regularization prevents overfitting

**Pipeline:**
```
Input Features → Preprocessing (Imputation + Encoding) 
→ Logistic Regression → Probability Calibration 
→ Threshold Optimization → Risk Scoring
```

### 4. Probability Calibration

**Why?** Raw model probabilities aren't always reliable for business decisions.

**Method:** Isotonic regression via cross-validation (`CalibratedClassifierCV`)

**Result:** When the model predicts 40% churn risk, ~40% of such customers actually churn.

### 5. Threshold Optimization

Instead of using 0.5 threshold, we optimize for **Expected Value**:

```python
Expected Value = (Churn Probability × Save Rate × LTV) - Offer Cost
              = (p × 0.30 × $800) - $100
```

**Optimal Threshold:** 46% churn probability
- Below 46%: Offer cost exceeds expected value (don't target)
- Above 46%: Positive expected value (target for retention)

### 6. Feature Importance

**Top Predictors** (via Permutation Importance):
1. Contract Type (Month-to-month)
2. Tenure
3. Internet Service Type
4. Tech Support availability
5. Online Security availability

## 📊 Dashboard & Visualizations

### Tableau Dashboard
Interactive dashboard available on Tableau Public: [View Dashboard](#)

**Features:**
- Executive KPI overview (churn rate, revenue at risk)
- Customer risk segmentation (High/Medium/Low)
- ML model predictions distribution
- Top customers to target for retention
- Segment analysis by contract, tenure, services

### Key Visualizations
1. **Churn Rate by Contract Type** - Horizontal bar chart
2. **Churn Trend by Tenure** - Line chart showing risk over time
3. **Risk Band Distribution** - Customer segmentation
4. **Churn Probability Histogram** - Model prediction distribution
5. **Top 20 Customers to Target** - Actionable customer list

## 💼 Business Recommendations

### Immediate Actions (High Priority)

1. **Deploy Targeted Retention Campaign**
   - Target: Customers with churn probability ≥46%
   - Action: Offer $100 retention incentive
   - Expected Impact: Prevent ~XXX churns, generate $XXX,XXX value

2. **Contract Migration Program**
   - Offer: 2 months free for switching to one-year contract
   - Target: Month-to-month customers
   - Expected Impact: 20-30% churn reduction if 50% convert

3. **New Customer Onboarding Blitz**
   - Focus: First 90 days (highest risk period)
   - Actions: Welcome calls, proactive support, usage tips
   - Expected Impact: 15% reduction in first-year churn

### Strategic Initiatives (Medium Priority)

4. **Service Bundle Strategy**
   - Create "Complete Care" bundles (Security + Support + Protection)
   - Offer 15% discount on bundles
   - Expected Impact: Increase attachment rate from 30% to 50%

5. **Fiber Service Quality Investigation**
   - Survey fiber customers on satisfaction
   - Benchmark pricing vs competitors
   - Audit service quality metrics

6. **Payment Method Optimization**
   - Incentivize autopay enrollment ($5/month discount)
   - Consider surcharge for electronic checks
   - Expected Impact: 40% shift to autopay, ~5% churn reduction

## 📊 Results & Model Performance

### Classification Metrics
- **ROC-AUC:** 0.84 (Excellent discrimination)
- **PR-AUC:** 0.65 (Good for imbalanced data)
- **Calibration:** Well-calibrated probabilities

### Business Metrics
- **Customers to Target:** ~XXX (update with actual)
- **Campaign Cost:** $XX,XXX
- **Expected Value:** $XXX,XXX
- **Net Benefit:** $XXX,XXX annually

### Segment Performance
Model performs consistently across customer segments (contract types, internet service, demographics).

## 📁 Files & Outputs

### Key Files

**`CustomerChurn.ipynb`**
- Complete ML pipeline from data loading to model deployment
- Includes EDA, modeling, calibration, threshold optimization
- Well-documented with markdown cells explaining each step

**`scored_customers_calibrated.csv`**
- Predictions for test set customers
- Columns: customerID, churn_probability, risk_band, target_flag, expected_value

**`customer_churn_documentation.md`**
- Detailed project writeup
- Business context, methodology, results, recommendations
- 30+ page comprehensive documentation

**`Churn_ML_Dashboard.twb`**
- Tableau workbook file
- Interactive dashboard with 6 sheets
- Requires Tableau Desktop to open

## 🛠️ Technical Stack

**Languages & Tools:**
- Python 3.10
- Jupyter Notebook
- Tableau Desktop Public Edition

**Libraries:**
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
```

**Machine Learning:**
- Logistic Regression
- Probability Calibration (Isotonic Regression)
- Permutation Importance
- Cross-validation

## 📈 Future Enhancements

1. **Model Comparison:** Test ensemble methods (Random Forest, XGBoost, Gradient Boosting)
2. **Feature Engineering:** Create interaction features, tenure bins, service combinations
3. **Deep Learning:** Experiment with neural networks for non-linear patterns
4. **Time Series:** Predict time-to-churn using survival analysis
5. **A/B Testing:** Deploy and measure actual retention campaign results
6. **Real-time Scoring:** Build API for live customer risk scoring

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Paramdeep Nijjer**
- GitHub: [@paramdeepnijjer-bliip](https://github.com/paramdeepnijjer-bliip)
- LinkedIn: [Your LinkedIn](#)
- Portfolio: [Your Portfolio](#)

## 🙏 Acknowledgments

- Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle
- Inspiration: Real-world customer retention challenges in telecommunications
- Tools: scikit-learn, Tableau, Jupyter

## 📞 Contact

For questions or collaborations, please reach out via:
- Email: your.email@example.com
- LinkedIn: [Your Profile](#)

---

**Star ⭐ this repository if you found it helpful!**

---

## 🖼️ Screenshots

### Dashboard Preview
![Customer Churn Dashboard](screenshots/dashboard.png)

### Model Results
![Model Performance](screenshots/model_performance.png)

### Key Insights
![EDA Insights](screenshots/eda_insights.png)

---

*Last Updated: February 2026*
