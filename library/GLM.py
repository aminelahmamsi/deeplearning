import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib.pyplot as plt

def run_full_glm_analysis(
        csv_path,
        response,
        factors,
        use_powers=False,
        interactions=True,
        transform_logit=False,
        remove_outliers=False
    ):
    """
    Perform a full General Linear Model experiment analysis:
    - Preprocessing (Outlier removal, Logit transform)
    - Fit GLM
    - Validate residuals (normality + homoscedasticity)
    - Produce ANOVA table
    - Plot residual diagnostics
    """

    df = pd.read_csv(csv_path)

    print(f"--- Data Loading: {len(df)} samples loaded ---")

    # --- 1. Outlier Removal (Clean Failed Runs) -------------------------------
    if remove_outliers:
        # We assume outliers are on the lower end (failed trainings)
        limit = df[response].mean() - 2 * df[response].std()
        original_count = len(df)
        df = df[df[response] > limit]
        print(f"-> Outlier Removal: Removed {original_count - len(df)} samples (value < {limit:.2f})")

    # --- 2. Logit Transformation (Fix Ceiling Effect) -------------------------
    target_col = response
    if transform_logit:
        print("-> Transformation: Applying Logit Transform (log(p / (1-p)))")
        # Normalize to 0-1 if data is 0-100
        p = df[response].copy()
        if p.max() > 1.0:
            p = p / 100.0
        
        # Clip to avoid infinity (log(0))
        epsilon = 1e-4
        p = p.clip(epsilon, 1 - epsilon)
        
        target_col = f"logit_{response}"
        df[target_col] = np.log(p / (1 - p))

    # --- 3. Build Formula -----------------------------------------------------
    numeric_factors = [f for f in factors if pd.api.types.is_numeric_dtype(df[f])]
    categorical_factors = [f for f in factors if f not in numeric_factors]
    
    # Base terms
    terms = categorical_factors + numeric_factors
    
    # Quadratic terms (Polynomials)
    if use_powers:
        terms += [f"I({f}**2)" for f in numeric_factors]
        
    # Construct formula string
    if interactions:
        rhs = " * ".join(categorical_factors + numeric_factors)
        if use_powers:
             rhs += " + " + " + ".join([f"I({f}**2)" for f in numeric_factors])
    else:
        rhs = " + ".join(terms)

    formula = f"{target_col} ~ {rhs}"

    print("\n=== MODEL FORMULA ===")
    print(formula)

    # --- Fit the model --------------------------------------------------------
    model = smf.ols(formula=formula, data=df).fit()

    print("\n=== MODEL SUMMARY ===")
    print(model.summary())

    # --- ANOVA table ----------------------------------------------------------
    print("\n=== ANOVA TABLE (Factor Significance) ===")
    try:
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)
    except Exception as e:
        print("Could not generate ANOVA table (likely due to perfect fit or singular matrix).")
        anova_table = None

    # --- Residual diagnostics -------------------------------------------------
    residuals = model.resid
    fitted = model.fittedvalues

    # Normality test (Shapiro–Wilk)
    shapiro_p = stats.shapiro(residuals)[1]
    print(f"\nNormality of residuals (Shapiro–Wilk p-value): {shapiro_p:.2e}")
    if shapiro_p >= 0.05:
        print("✓ Residuals are normal")
    else:
        print("✗ Residuals deviate from normality")

    # Homoscedasticity test (Breusch–Pagan)
    try:
        bp_p = sm.stats.diagnostic.het_breuschpagan(residuals, model.model.exog)[1]
        print(f"Homoscedasticity (Breusch–Pagan p-value): {bp_p:.2e}")
        if bp_p >= 0.05:
            print("✓ Variances are homogeneous")
        else:
            print("✗ Heteroscedasticity detected")
    except:
        print("Could not run Breusch-Pagan test.")

    # --- Diagnostic plots ------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # QQ-plot
    sm.qqplot(residuals, line='45', fit=True, ax=axs[0])
    axs[0].set_title("Normal Probability Plot (QQ-Plot)")

    # Residuals vs Fitted
    axs[1].scatter(fitted, residuals, alpha=0.6)
    axs[1].axhline(0, color='red', linestyle='--')
    axs[1].set_xlabel("Fitted values")
    axs[1].set_ylabel("Residuals")
    axs[1].set_title("Residuals vs Fitted")

    plt.tight_layout()
    plt.show()

    return model, anova_table