📉 Customer Churn Prediction & Retention Optimization
A business-focused machine learning project that predicts telecom customer churn and optimizes retention strategy using probability calibration and threshold-based ROI maximization.
ROC-AUC: 0.84  |  Optimal Threshold: ~42%  |  Max Expected Value: $18,830 on held-out test set
🚀 Live Demo →

🧠 The Business Problem
A telecom company loses significant revenue every time a customer cancels. The challenge isn't just predicting who will churn — it's deciding who to target with a retention offer in a way that's actually profitable.
Targeting everyone is wasteful. Targeting only the highest-risk customers misses the recoverable middle. This project builds a model that answers: "Given this customer's churn probability, does it make financial sense to send them a retention offer?"
Business assumptions:

Customer Lifetime Value: $800
Retention offer cost: $100 per customer
Offer success rate: 30% (customers who accept and stay)
Goal: maximize Expected Value = (churn_prob × save_rate × LTV) − offer_cost


📊 Dataset
IBM Telco Customer Churn — 7,043 customers, 21 features
Features include customer demographics, account information (contract type, tenure, payment method), services subscribed (phone, internet, add-ons), and monthly/total charges.
Target variable: Churn (Yes/No) — 26.5% churn rate (class imbalance addressed with class_weight='balanced')

🔍 Key EDA Findings
DriverFindingContract TypeMonth-to-month customers churn at 42% vs 3% for two-year contractsTenureFirst 12 months have >50% churn — drops sharply after year oneInternet ServiceFiber optic customers churn ~30-35%, DSL ~20%Service BundlesCustomers without OnlineSecurity or TechSupport churn significantly morePayment MethodElectronic check users churn 3x more than autopay usersMonthly ChargesHigher charges ($70–110 range) correlate with higher churn risk

⚙️ Modeling Pipeline
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
Why Logistic Regression?
Interpretable, probability-native, and a strong baseline for a business decision problem. The goal is reliable probability estimates — not squeezing out marginal accuracy from a black-box model.
Why calibrate?
Raw model probabilities aren't reliable for threshold-based decisions. Isotonic calibration reduced the Brier Score from 0.169 → 0.139, ensuring "40% predicted churn" actually corresponds to ~40% observed churn.

📈 Results
MetricUncalibratedCalibratedROC-AUC0.8420.841PR-AUC0.6330.628Brier Score0.1690.139 ✅
Threshold Optimization:

Optimal threshold: ~42% churn probability
Customers targeted at this threshold: 401
Max Expected Value on test set: $18,830

Segment-level performance:
Contract TypeNChurn RateROC-AUCMonth-to-month77342.6%0.741One year30012.0%0.746Two year3362.7%0.765

💡 Business Recommendations

Contract migration is the highest-ROI lever — converting month-to-month customers to annual contracts is worth prioritizing over blanket discounts
Onboarding investment in year one — churn risk is highest in the first 12 months; front-loading engagement here pays off
Bundle cross-selling — customers without OnlineSecurity and TechSupport churn far more; targeted bundle offers serve both retention and revenue goals
Payment method nudge — electronic check users churn 3x more; nudging toward auto-pay is low-cost and high-signal


📁 Repository Structure
├── app.py                    # Streamlit web app (live demo)
├── CustomerChurn.ipynb       # Full end-to-end analysis and modeling notebook
├── Telco-Customer-Churn.csv  # Source dataset
├── requirements.txt          # Python dependencies
└── README.md

🛠️ Tech Stack
Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Streamlit
Key techniques: Pipeline + ColumnTransformer, CalibratedClassifierCV (isotonic), permutation importance, threshold optimization via expected value maximization, segment-level model evaluation

🚀 How to Run Locally
bashgit clone https://github.com/paramdeepnijjer-blip/Customer-Churn-ML
cd Customer-Churn-ML
pip install -r requirements.txt

# Run the web app
streamlit run app.py

# Or open the notebook
jupyter notebook CustomerChurn.ipynb

👤 Author
Paramdeep Nijjer — M.S. Data Science, Boston University
LinkedIn · GitHub · Live Demo
