"""
Lab-4: Interactive Statistical Modeling Dashboard
Dataset: Medical Insurance Costs (insurance.csv)

Run with:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

# ---------------------------------------------------------------------------
# Page config & data loading
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Medical Insurance Cost Dashboard",
    page_icon="💊",
    layout="wide",
)

NUMERIC_COLS = ["age", "bmi", "children", "charges"]
CATEGORICAL_COLS = ["sex", "smoker", "region"]

# Anchor the data path to this script's own folder, NOT the current working
# directory. Streamlit Cloud runs the app with cwd = repo root, which breaks
# plain relative paths like "data/insurance.csv" whenever app.py lives in a
# subfolder of the repo.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "insurance.csv")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource
def fit_model(df: pd.DataFrame):
    """Fit the OLS model used across the dashboard (Tab 3)."""
    formula = "charges ~ age + bmi + children + C(sex) + C(smoker) + C(region) + bmi:C(smoker)"
    model = smf.ols(formula=formula, data=df).fit()
    return model


df_full = load_data()
model = fit_model(df_full)

st.title("💊 Medical Insurance Cost — Statistical Dashboard")
st.caption(
    "M.Sc. Data Science · Lab-4 · Applied Statistical Modeling & Interactive Web Dashboard"
)

tab1, tab2, tab3 = st.tabs(
    ["📊 Data Exploration", "🧪 Hypothesis Testing Lab", "🔮 Live Prediction & Diagnostics"]
)

# ---------------------------------------------------------------------------
# TAB 1 — DATA EXPLORATION
# ---------------------------------------------------------------------------
with tab1:
    st.header("Data Exploration")

    with st.sidebar:
        st.subheader("🔎 Filters")
        age_range = st.slider(
            "Age range",
            int(df_full["age"].min()), int(df_full["age"].max()),
            (int(df_full["age"].min()), int(df_full["age"].max())),
        )
        bmi_range = st.slider(
            "BMI range",
            float(df_full["bmi"].min()), float(df_full["bmi"].max()),
            (float(df_full["bmi"].min()), float(df_full["bmi"].max())),
        )
        regions = st.multiselect(
            "Region", options=sorted(df_full["region"].unique()),
            default=sorted(df_full["region"].unique()),
        )
        smoker_sel = st.multiselect(
            "Smoker status", options=sorted(df_full["smoker"].unique()),
            default=sorted(df_full["smoker"].unique()),
        )
        sex_sel = st.multiselect(
            "Sex", options=sorted(df_full["sex"].unique()),
            default=sorted(df_full["sex"].unique()),
        )

    mask = (
        df_full["age"].between(*age_range)
        & df_full["bmi"].between(*bmi_range)
        & df_full["region"].isin(regions)
        & df_full["smoker"].isin(smoker_sel)
        & df_full["sex"].isin(sex_sel)
    )
    df = df_full[mask]

    st.markdown(f"**{len(df):,} of {len(df_full):,} records** match the current filters.")

    if df.empty:
        st.warning("No records match the current filter selection. Widen a filter.")
    else:
        # Summary stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg. Charges", f"${df['charges'].mean():,.0f}")
        c2.metric("Median Charges", f"${df['charges'].median():,.0f}")
        c3.metric("Avg. BMI", f"{df['bmi'].mean():.1f}")
        c4.metric("Smoker %", f"{(df['smoker'] == 'yes').mean() * 100:.1f}%")

        st.dataframe(df.describe().round(2), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(
                df, x="charges", color="smoker", marginal="box", nbins=40,
                title="Distribution of Charges (by smoker status)",
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        with col2:
            fig_scatter = px.scatter(
                df, x="bmi", y="charges", color="smoker", size="age",
                hover_data=["age", "region", "children"],
                title="BMI vs Charges (bubble size = age)",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig_box = px.box(
                df, x="region", y="charges", color="smoker",
                title="Charges by Region and Smoker Status",
            )
            st.plotly_chart(fig_box, use_container_width=True)
        with col4:
            corr = df[NUMERIC_COLS].corr()
            fig_corr = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Correlation Matrix",
            )
            st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 — HYPOTHESIS TESTING LAB
# ---------------------------------------------------------------------------
with tab2:
    st.header("Hypothesis Testing Lab")
    st.write(
        "Pick a categorical factor and a numerical metric. The app automatically "
        "chooses the right test, runs it, and reports the conclusion at α = 0.05."
    )

    colA, colB = st.columns(2)
    with colA:
        factor = st.selectbox("Categorical factor", CATEGORICAL_COLS, index=1)
    with colB:
        metric = st.selectbox("Numerical metric", [c for c in NUMERIC_COLS if c != "children"] + ["children"])

    groups_dict = {level: g[metric].values for level, g in df_full.groupby(factor)}
    n_groups = len(groups_dict)
    alpha = 0.05

    st.subheader(f"Testing: does **{metric}** differ across levels of **{factor}**?")

    if n_groups == 2:
        (g1_name, g1), (g2_name, g2) = list(groups_dict.items())
        st.markdown(f"**H0:** Mean {metric} is equal for `{g1_name}` and `{g2_name}`.  \n"
                    f"**H1:** Mean {metric} differs between `{g1_name}` and `{g2_name}`.")

        sw1_p = stats.shapiro(g1).pvalue if len(g1) >= 3 else np.nan
        sw2_p = stats.shapiro(g2).pvalue if len(g2) >= 3 else np.nan
        lev_p = stats.levene(g1, g2).pvalue
        normal = (sw1_p > alpha) and (sw2_p > alpha)
        equal_var = lev_p > alpha

        if normal:
            stat, p_val = stats.ttest_ind(g1, g2, equal_var=equal_var)
            test_name = "Two-Sample t-test" + (" (Welch)" if not equal_var else "")
        else:
            stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            test_name = "Mann-Whitney U test"

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Shapiro-Wilk p ({g1_name})", f"{sw1_p:.4g}")
        m2.metric(f"Shapiro-Wilk p ({g2_name})", f"{sw2_p:.4g}")
        m3.metric("Levene's p-value", f"{lev_p:.4g}")

        st.info(f"Normality assumption met: **{normal}** → Running **{test_name}**")

    elif n_groups > 2:
        st.markdown(f"**H0:** Mean {metric} is equal across all levels of `{factor}`.  \n"
                    f"**H1:** At least one group's mean {metric} differs.")
        stat, p_val = stats.f_oneway(*groups_dict.values())
        test_name = "One-Way ANOVA"
    else:
        st.warning("The selected factor needs at least 2 levels to run a test.")
        st.stop()

    r1, r2 = st.columns(2)
    r1.metric("Test statistic", f"{stat:.4f}")
    r2.metric("p-value", f"{p_val:.4g}")

    conclusion = "Reject H0 ❌" if p_val < alpha else "Fail to Reject H0 ✅"
    color = "red" if p_val < alpha else "green"
    st.markdown(f"### Conclusion ({test_name}) at α = 0.05: :{color}[{conclusion}]")

    fig_box2 = px.box(df_full, x=factor, y=metric, points="all", title=f"{metric} by {factor}")
    st.plotly_chart(fig_box2, use_container_width=True)

    st.divider()
    st.subheader("Bonus: Chi-Square test between two categorical variables")
    colC, colD = st.columns(2)
    with colC:
        cat1 = st.selectbox("Categorical variable 1", CATEGORICAL_COLS, index=1, key="cat1")
    with colD:
        cat2 = st.selectbox("Categorical variable 2", CATEGORICAL_COLS, index=2, key="cat2")

    if cat1 == cat2:
        st.warning("Choose two different categorical variables.")
    else:
        contingency = pd.crosstab(df_full[cat1], df_full[cat2])
        chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
        st.write("Contingency table:")
        st.dataframe(contingency)
        st.write(f"Chi² = {chi2:.4f}, dof = {dof}, p-value = {chi_p:.4g}")
        chi_conclusion = "Reject H0 ❌ (associated)" if chi_p < alpha else "Fail to Reject H0 ✅ (independent)"
        st.markdown(f"**Conclusion:** {chi_conclusion}")

# ---------------------------------------------------------------------------
# TAB 3 — LIVE PREDICTION & DIAGNOSTICS
# ---------------------------------------------------------------------------
with tab3:
    st.header("Live Prediction & Model Diagnostics")
    st.caption(
        "Model: `charges ~ age + bmi + children + C(sex) + C(smoker) + C(region) + bmi:C(smoker)`"
    )

    st.subheader("🔮 Predict charges for a new individual")
    p1, p2, p3 = st.columns(3)
    with p1:
        in_age = st.slider("Age", 18, 64, 35)
        in_bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.0, step=0.1)
    with p2:
        in_children = st.slider("Children", 0, 5, 0)
        in_sex = st.selectbox("Sex", sorted(df_full["sex"].unique()))
    with p3:
        in_smoker = st.selectbox("Smoker", sorted(df_full["smoker"].unique()))
        in_region = st.selectbox("Region", sorted(df_full["region"].unique()))

    new_point = pd.DataFrame([{
        "age": in_age, "bmi": in_bmi, "children": in_children,
        "sex": in_sex, "smoker": in_smoker, "region": in_region,
    }])

    pred = model.get_prediction(new_point)
    pred_summary = pred.summary_frame(alpha=0.05)
    point_pred = pred_summary["mean"].iloc[0]
    ci_low, ci_high = pred_summary["mean_ci_lower"].iloc[0], pred_summary["mean_ci_upper"].iloc[0]
    pi_low, pi_high = pred_summary["obs_ci_lower"].iloc[0], pred_summary["obs_ci_upper"].iloc[0]

    m1, m2 = st.columns(2)
    m1.metric("Predicted Charges", f"${point_pred:,.2f}")
    m2.metric("95% CI (mean response)", f"${ci_low:,.0f} – ${ci_high:,.0f}")
    st.write(f"95% Prediction Interval (individual): **${pi_low:,.0f} – ${pi_high:,.0f}**")

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=df_full["bmi"], y=df_full["charges"], mode="markers",
        marker=dict(color=df_full["smoker"].map({"yes": "crimson", "no": "steelblue"}), opacity=0.4),
        name="Observed data",
    ))
    fig_pred.add_trace(go.Scatter(
        x=[in_bmi], y=[point_pred], mode="markers",
        marker=dict(color="gold", size=16, symbol="star", line=dict(color="black", width=1)),
        name="Your prediction",
    ))
    fig_pred.update_layout(
        title="Your prediction in context (BMI vs Charges)",
        xaxis_title="BMI", yaxis_title="Charges",
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    st.divider()
    st.subheader("📐 Residual Diagnostics (Gauss-Markov checks)")

    fitted = model.fittedvalues
    resid = model.resid

    d1, d2 = st.columns(2)
    with d1:
        fig_resid = px.scatter(
            x=fitted, y=resid, opacity=0.5,
            labels={"x": "Fitted values", "y": "Residuals"},
            title="Residuals vs Fitted Values",
        )
        fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_resid, use_container_width=True)

    with d2:
        qq = sm.ProbPlot(resid)
        theoretical_q = qq.theoretical_quantiles
        sample_q = qq.sample_quantiles
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=theoretical_q, y=sample_q, mode="markers", name="Residuals"))
        line_x = np.array([theoretical_q.min(), theoretical_q.max()])
        fig_qq.add_trace(go.Scatter(
            x=line_x, y=line_x * resid.std() + resid.mean(),
            mode="lines", name="Reference line", line=dict(color="red", dash="dash"),
        ))
        fig_qq.update_layout(title="Q-Q Plot of Residuals",
                              xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
        st.plotly_chart(fig_qq, use_container_width=True)

    jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(resid)
    j1, j2, j3 = st.columns(3)
    j1.metric("Jarque-Bera stat", f"{jb_stat:.3f}")
    j2.metric("JB p-value", f"{jb_p:.4g}")
    j3.metric("R² / Adj. R²", f"{model.rsquared:.3f} / {model.rsquared_adj:.3f}")

    st.write("**Variance Inflation Factor (continuous predictors)**")
    X_vif = sm.add_constant(df_full[["age", "bmi", "children"]])
    vif_df = pd.DataFrame({
        "feature": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
    })
    st.dataframe(vif_df.round(3), use_container_width=True)

    with st.expander("Show full OLS regression summary"):
        st.text(model.summary().as_text())
