"""
Lab-4: Applied Statistical Modeling
Part 1 - Exploratory Data Analysis & Hypothesis Testing
Part 2 - OLS Regression & Gauss-Markov Diagnostics

Dataset: Medical Insurance Costs (insurance.csv)
Columns: age, sex, bmi, children, smoker, region, charges

Run with:  python analysis/eda_and_modeling.py
All plots are saved to analysis/plots/. All numeric/test output is printed
to stdout (redirect to a .txt file if you want it captured for your README).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

sns.set_theme(style="whitegrid")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "insurance.csv")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# PART 1: EDA & HYPOTHESIS TESTING
# ---------------------------------------------------------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def descriptive_stats(df):
    section("1.1 DESCRIPTIVE METRICS (numerical features)")
    num_cols = ["age", "bmi", "children", "charges"]
    desc = df[num_cols].describe().T
    desc["iqr"] = df[num_cols].quantile(0.75) - df[num_cols].quantile(0.25)
    desc["skewness"] = df[num_cols].skew()
    desc["kurtosis"] = df[num_cols].kurt()
    print(desc.round(3))
    return desc


def visual_exploration(df):
    section("1.2 VISUAL EXPLORATION")
    num_cols = ["age", "bmi", "children", "charges"]

    # Distribution plots (hist + KDE)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.ravel(), num_cols):
        sns.histplot(df[col], kde=True, ax=ax, color="#3b6ea5")
        ax.set_title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "distributions.png"), dpi=150)
    plt.close()

    # Correlation matrix
    corr = df[num_cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Matrix (numerical features)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "correlation_matrix.png"), dpi=150)
    plt.close()

    # Bivariate scatter: bmi vs charges colored by smoker; age vs charges
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker", ax=axes[0])
    axes[0].set_title("BMI vs Charges (by smoker status)")
    sns.scatterplot(data=df, x="age", y="charges", hue="smoker", ax=axes[1])
    axes[1].set_title("Age vs Charges (by smoker status)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "bivariate_scatter.png"), dpi=150)
    plt.close()

    print(f"Saved: distributions.png, correlation_matrix.png, bivariate_scatter.png -> {PLOT_DIR}")
    print("\nCorrelation matrix:\n", corr.round(3))


def hypothesis_test_1_smoker_charges(df):
    """Two-group comparison: Smokers vs Non-Smokers on Charges."""
    section("1.3 HYPOTHESIS TEST 1: Smokers vs Non-Smokers -> Charges")
    print("H0: There is no difference in mean charges between smokers and non-smokers.")
    print("H1: There is a significant difference in mean charges between smokers and non-smokers.")

    grp_smoker = df.loc[df["smoker"] == "yes", "charges"]
    grp_nonsmoker = df.loc[df["smoker"] == "no", "charges"]

    # Normality check (Shapiro-Wilk) on each group
    sw_smoker = stats.shapiro(grp_smoker)
    sw_nonsmoker = stats.shapiro(grp_nonsmoker)
    print(f"\nShapiro-Wilk (smokers):     W={sw_smoker.statistic:.4f}, p={sw_smoker.pvalue:.4g}")
    print(f"Shapiro-Wilk (non-smokers): W={sw_nonsmoker.statistic:.4f}, p={sw_nonsmoker.pvalue:.4g}")
    normal = sw_smoker.pvalue > 0.05 and sw_nonsmoker.pvalue > 0.05

    # Equal variance check (Levene's test)
    levene_stat, levene_p = stats.levene(grp_smoker, grp_nonsmoker)
    print(f"Levene's test:              stat={levene_stat:.4f}, p={levene_p:.4g}")
    equal_var = levene_p > 0.05

    if normal:
        t_stat, p_val = stats.ttest_ind(grp_smoker, grp_nonsmoker, equal_var=equal_var)
        test_name = "Two-Sample t-test" + (" (equal var)" if equal_var else " (Welch, unequal var)")
        stat_val = t_stat
    else:
        stat_val, p_val = stats.mannwhitneyu(grp_smoker, grp_nonsmoker, alternative="two-sided")
        test_name = "Mann-Whitney U test"

    print(f"\nNormality assumption met: {normal}  ->  Running: {test_name}")
    print(f"Test statistic = {stat_val:.4f}, p-value = {p_val:.4g}")

    alpha = 0.05
    conclusion = "Reject H0" if p_val < alpha else "Fail to Reject H0"
    print(f"\nConclusion at alpha=0.05: {conclusion} "
          f"-> {'Smokers and non-smokers have significantly different mean charges.' if p_val < alpha else 'No significant difference detected.'}")
    return {
        "test_name": test_name, "statistic": stat_val, "p_value": p_val,
        "normal": normal, "equal_var": equal_var, "conclusion": conclusion,
    }


def hypothesis_test_2_chi_square(df):
    """Chi-Square: is smoking status linked to region?"""
    section("1.4 HYPOTHESIS TEST 2 (Option A): Chi-Square - Smoker vs Region")
    print("H0: Smoking status is independent of region.")
    print("H1: Smoking status is associated with region.")

    contingency = pd.crosstab(df["smoker"], df["region"])
    print("\nContingency table:\n", contingency)

    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    print(f"\nChi2 = {chi2:.4f}, dof = {dof}, p-value = {p_val:.4g}")

    alpha = 0.05
    conclusion = "Reject H0" if p_val < alpha else "Fail to Reject H0"
    print(f"Conclusion at alpha=0.05: {conclusion} "
          f"-> {'Smoking status and region are associated.' if p_val < alpha else 'No significant association detected.'}")
    return {"chi2": chi2, "dof": dof, "p_value": p_val, "conclusion": conclusion}


def hypothesis_test_2_anova(df):
    """One-Way ANOVA: do average charges differ across regions?"""
    section("1.4 HYPOTHESIS TEST 2 (Option B): One-Way ANOVA - Charges across Regions")
    print("H0: Mean charges are equal across all four regions.")
    print("H1: At least one region's mean charges differ.")

    groups = [g["charges"].values for _, g in df.groupby("region")]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\nF-statistic = {f_stat:.4f}, p-value = {p_val:.4g}")

    alpha = 0.05
    conclusion = "Reject H0" if p_val < alpha else "Fail to Reject H0"
    print(f"Conclusion at alpha=0.05: {conclusion} "
          f"-> {'At least one region differs significantly.' if p_val < alpha else 'No significant difference across regions.'}")
    return {"f_stat": f_stat, "p_value": p_val, "conclusion": conclusion}


# ---------------------------------------------------------------------------
# PART 2: OLS REGRESSION & DIAGNOSTICS
# ---------------------------------------------------------------------------

def fit_ols_model(df):
    section("2.1 MODEL FORMULATION: OLS Regression")
    print("Model: charges ~ age + bmi + children + C(sex) + C(smoker) + C(region) + bmi:C(smoker)")

    formula = "charges ~ age + bmi + children + C(sex) + C(smoker) + C(region) + bmi:C(smoker)"
    model = smf.ols(formula=formula, data=df).fit()
    print(model.summary())
    return model


def interpret_model(model):
    section("2.2 PARAMETER INTERPRETATION")
    params = model.params
    conf = model.conf_int()
    conf.columns = ["CI_lower_2.5%", "CI_upper_97.5%"]
    summary_tbl = pd.concat([params.rename("coef"), model.pvalues.rename("p_value"), conf], axis=1)
    print(summary_tbl.round(4))
    print(f"\nR-squared:          {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic p-value: {model.f_pvalue:.4g}")
    return summary_tbl


def gauss_markov_diagnostics(df, model):
    section("2.3 GAUSS-MARKOV DIAGNOSTIC CHECKS")
    fitted = model.fittedvalues
    resid = model.resid

    # --- Linearity & Homoscedasticity: Residuals vs Fitted ---
    plt.figure(figsize=(7, 5))
    plt.scatter(fitted, resid, alpha=0.5, color="#3b6ea5")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted Values")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "residuals_vs_fitted.png"), dpi=150)
    plt.close()

    # --- Normality of residuals: Q-Q plot ---
    fig = sm.qqplot(resid, line="s")
    fig.set_size_inches(6, 5)
    plt.title("Q-Q Plot of Residuals")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "qq_plot_residuals.png"), dpi=150)
    plt.close()

    jb_stat, jb_p, skew, kurt = jarque_bera(resid)
    print(f"Jarque-Bera test: JB={jb_stat:.4f}, p={jb_p:.4g}, skew={skew:.4f}, kurtosis={kurt:.4f}")
    print("(Omnibus / Prob(Omnibus) values are also reported directly in the OLS summary table above.)")

    # --- Multicollinearity: VIF ---
    section("2.3b MULTICOLLINEARITY: Variance Inflation Factor (continuous predictors)")
    X = df[["age", "bmi", "children"]].copy()
    X = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_data.round(3))

    print(f"\nSaved: residuals_vs_fitted.png, qq_plot_residuals.png -> {PLOT_DIR}")
    return {"jb_stat": jb_stat, "jb_p": jb_p, "vif": vif_data}


def main():
    df = load_data()
    section("DATA LOADED")
    print(df.head())
    print(f"\nShape: {df.shape}")

    descriptive_stats(df)
    visual_exploration(df)
    hypothesis_test_1_smoker_charges(df)
    hypothesis_test_2_chi_square(df)
    hypothesis_test_2_anova(df)

    model = fit_ols_model(df)
    interpret_model(model)
    gauss_markov_diagnostics(df, model)

    section("DONE - all plots saved in analysis/plots/")


if __name__ == "__main__":
    main()
