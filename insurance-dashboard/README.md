# Medical Insurance Cost — Statistical Modeling & Interactive Dashboard

**Course:** Statistical Modeling with Python (M.Sc. Data Science, Sem 1) · Lab-4
**Dataset:** Medical Insurance Costs (`insurance.csv`, Kaggle "Medical Cost Personal Datasets")

---

## 1. Dataset Summary

| Field | Description |
|---|---|
| `age` | Age of primary beneficiary (18–64) |
| `sex` | `male` / `female` |
| `bmi` | Body Mass Index |
| `children` | Number of dependents covered |
| `smoker` | `yes` / `no` |
| `region` | `northeast`, `northwest`, `southeast`, `southwest` (US) |
| `charges` | Individual medical costs billed by insurance (target variable) |

- **1,338 rows**, no missing values.
- `charges` is strongly right-skewed (skewness ≈ 1.52); `age` and `bmi` are close to symmetric.

## 2. How to Run

### Option A — Local (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Run the standalone analysis script (Part 1 & 2 — prints stats, saves plots to analysis/plots/)
python analysis/eda_and_modeling.py

# 2. Launch the interactive dashboard (Part 3)
streamlit run app.py            # opens at http://localhost:8501
```

### Option B — Google Colab
```python
!pip install streamlit
%%writefile app.py
# (paste app.py contents)
!streamlit run app.py & npx localtunnel --port 8501
```

## 3. Project Structure
```
insurance-dashboard/
├── app.py                        # Streamlit dashboard (3 tabs)
├── requirements.txt
├── README.md
├── data/
│   └── insurance.csv
└── analysis/
    ├── eda_and_modeling.py       # Part 1 (EDA + hypothesis tests) & Part 2 (OLS + diagnostics)
    └── plots/                    # Saved output plots (generated on run)
```

## 4. Synthesis of Statistical Findings

### Part 1 — Hypothesis Testing

**Test 1 — Smokers vs. Non-Smokers on `charges`:**
- Shapiro-Wilk: smokers `p ≈ 3.6e-9`, non-smokers `p ≈ 1.4e-28` → **neither group is normally distributed**.
- Levene's test: `p ≈ 1.6e-66` → variances are also unequal.
- Since normality fails → used **Mann-Whitney U test**: `U ≈ 284,133`, `p ≈ 5.3e-130`.
- **Conclusion: Reject H0.** Smokers incur significantly higher medical charges than non-smokers — by a wide margin.

**Test 2 — Categorical association / group comparison:**
- *Chi-Square (smoker × region)*: `χ² ≈ 7.34`, dof = 3, `p ≈ 0.062` → **Fail to Reject H0** — no statistically significant association between smoking status and region at α = 0.05 (smoking rates are roughly similar across regions).
- *One-Way ANOVA (charges across 4 regions)*: `F ≈ 2.97`, `p ≈ 0.031` → **Reject H0** — average charges differ significantly by region (though the effect is modest compared to the smoker effect).

### Part 2 — OLS Regression

**Model:** `charges ~ age + bmi + children + C(sex) + C(smoker) + C(region) + bmi:C(smoker)`

- **R² = 0.841, Adjusted R² = 0.840** — the model explains ~84% of the variance in charges, a strong fit driven mainly by smoking status and its interaction with BMI.
- **Age**: +$263.62 per year (`p < 0.001`) — highly significant, steady cost increase with age.
- **Children**: +$516.40 per dependent (`p < 0.001`) — significant.
- **Sex (male vs. female)**: −$500.15, borderline (`p ≈ 0.061`) — not significant at α = 0.05.
- **Region**: Southeast and Southwest are significantly cheaper than the Northeast reference (`p ≈ 0.0016` and `0.0013`); Northwest is not significantly different (`p ≈ 0.12`).
- **Smoker × BMI interaction is the key story**: the `smoker_yes` main effect (−$20,415) and the `bmi:smoker_yes` interaction (+$1,443 per BMI unit) must be read together — they show that being a smoker on its own doesn't shift cost much at very low BMI, but **cost rises steeply with BMI *only* for smokers**. A smoker with BMI 30 costs roughly $20,000+ more than a similar non-smoker, which matches the well-documented compounding health risk of obesity + smoking.
- **BMI alone** (for non-smokers) is *not* significant (`p ≈ 0.358`) — its effect on cost is almost entirely mediated through the smoking interaction.

**Gauss-Markov Diagnostics:**
- *Linearity/Homoscedasticity*: Residuals-vs-fitted plot shows a funnel/curvature pattern — some heteroscedasticity is present, largely from the two-cluster (smoker/non-smoker) structure of the target.
- *Normality*: Jarque-Bera `p ≈ 0.0`, Shapiro-Wilk on residuals `p ≈ 0.0` — **residuals are not normally distributed** (right-skew persists, consistent with `charges` itself being skewed). A log-transform of `charges` would likely improve this — left as a suggested extension.
- *Multicollinearity*: VIF for `age` (1.01), `bmi` (1.01), `children` (1.00) — **all well below the common threshold of 5**, so multicollinearity is not a concern among the continuous predictors.

**Overall takeaway:** Smoking status (and its interaction with BMI) is by far the dominant driver of medical charges, followed by age. Region and sex play smaller, more marginal roles. The model fits well (R² ≈ 0.84) but violates the normality-of-residuals assumption, so p-values/CIs should be interpreted with some caution — a log(charges) transformation is a natural next step for a more textbook-compliant OLS fit.

## 5. Dashboard Overview (Part 3)

| Tab | Contents |
|---|---|
| **📊 Data Exploration** | Sidebar filters (age/BMI sliders, region/smoker/sex multiselects), reactive Plotly histograms/scatter/box/correlation plots, live summary metrics |
| **🧪 Hypothesis Testing Lab** | Dropdowns to pick any categorical factor + numeric metric → auto-selects t-test / Mann-Whitney / ANOVA, reports statistic, p-value, and Reject/Fail-to-Reject conclusion; plus an ad-hoc Chi-Square tool for any two categorical variables |
| **🔮 Live Prediction & Diagnostics** | Sliders/inputs for a new individual → real-time OLS prediction with 95% CI (mean) and prediction interval (individual), plus residuals-vs-fitted and Q-Q diagnostic plots, VIF table, and full model summary |

## 6. Optional Bonus
To claim the +5 mark bonus, deploy `app.py` to [Streamlit Community Cloud](https://share.streamlit.io) and add the public URL here:

> **Live demo:** _add your share.streamlit.io URL here after deploying_
