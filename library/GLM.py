import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pyplot as plt

def run_full_glm_analysis(
        csv_path,
        response,
        factors,
        interactions=True
    ):
    """
    Perform a full General Linear Model experiment analysis:
    - Fit GLM
    - Validate residuals (normality + homoscedasticity)
    - Produce ANOVA table
    - Plot residual diagnostics

    Parameters:
        csv_path (str): Path to CSV.
        response (str): Dependent variable to analyze.
        factors (list[str]): Independent variables.
        interactions (bool): Include all 2nd-order interactions.

    Returns:
        model: fitted GLM model
        anova_table: ANOVA significance table
    """

    df = pd.read_csv(csv_path)

    # --- Build formula --------------------------------------------------------
    if interactions:
        # Example: train_losses + dropout + learning_rate
        # → train_losses * dropout * learning_rate
        formula = response + " ~ " + " * ".join(factors)
    else:
        formula = response + " ~ " + " + ".join(factors)

    print("\n=== MODEL FORMULA ===")
    print(formula)

    # --- Fit the model --------------------------------------------------------
    model = smf.ols(formula=formula, data=df).fit()

    print("\n=== MODEL SUMMARY ===")
    print(model.summary())

    # --- ANOVA table ----------------------------------------------------------
    print("\n=== ANOVA TABLE (Factor Significance) ===")
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # --- Residual diagnostics -------------------------------------------------
    residuals = model.resid
    fitted = model.fittedvalues

    # Normality test (Shapiro–Wilk)
    shapiro_p = stats.shapiro(residuals)[1]
    print("\nNormality of residuals (Shapiro–Wilk p-value):", shapiro_p)
    if shapiro_p >= 0.05:
        print("✓ Residuals are normal")
    else:
        print("✗ Residuals deviate from normality")

    # Homoscedasticity test (Breusch–Pagan)
    bp_p = sm.stats.diagnostic.het_breuschpagan(residuals, model.model.exog)[1]
    print("Homoscedasticity (Breusch–Pagan p-value):", bp_p)
    if bp_p >= 0.05:
        print("✓ Variances are homogeneous")
    else:
        print("✗ Heteroscedasticity detected")

    # --- Diagnostic plots ------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # QQ-plot
    sm.qqplot(residuals, line='45', ax=axs[0])
    axs[0].set_title("Normal Probability Plot")

    # Residuals vs Fitted
    axs[1].scatter(fitted, residuals)
    axs[1].axhline(0, color='red')
    axs[1].set_xlabel("Fitted values")
    axs[1].set_ylabel("Residuals")
    axs[1].set_title("Residuals vs Fitted")

    plt.tight_layout()
    plt.show()

    return model, anova_table
