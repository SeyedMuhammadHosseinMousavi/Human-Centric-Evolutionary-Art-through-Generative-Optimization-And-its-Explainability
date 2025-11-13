# ============================================================
# PSO / HILL - EXPLAINABILITY SUITE (SHAP • LIME • PERM IMPORT)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import warnings

# ------------------------ CONFIG ------------------------
MODEL_PATH = "pso_human_multishape_metrics_model.npz"
sns.set(style="whitegrid", font_scale=1.2)
warnings.filterwarnings("ignore")

# ------------------------ LOAD ------------------------
print(f"Loading model: {MODEL_PATH}")
data = np.load(MODEL_PATH, allow_pickle=True)

# ---- Detect whether this is Hill-Climbing or PSO model ----
if "rectangles" in data:
    print("Detected Hill-Climbing archive format.")
    rectangles = np.array(data["rectangles"])
    delta_error = np.array(data["delta_error"])
    cols = ["x", "y", "w", "h", "angle", "R", "G", "B", "opacity"]
else:
    print("Detected PSO archive format.")
    rectangles = np.array(data["params"])
    delta_error = np.array(data["delta"])
    cols = ["x", "y", "w", "h", "angle", "R", "G", "B", "opacity"]

# ---- Prepare DataFrame ----
df = pd.DataFrame(rectangles, columns=cols)

# ---- Target: negative delta error (lower MSE is better) ----
y = -delta_error
X = df.values

# ------------------------ TRAIN MODEL ------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("\n✅ RandomForestRegressor trained for explainability")
print(f"R² (train): {model.score(X_train, y_train):.3f}")
print(f"R² (test):  {model.score(X_test, y_test):.3f}")

# ============================================================
# 1️⃣ SHAP ANALYSIS
# ============================================================
print("\n🔍 SHAP Analysis: Explaining global + local parameter influence...")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# ---- Summary Bar Plot ----
plt.figure(figsize=(8,5))
shap.summary_plot(shap_values, X_test, feature_names=cols, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Global)")
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=300)
plt.show()

# ---- Beeswarm ----
shap.summary_plot(shap_values, X_test, feature_names=cols, show=False)
plt.title("SHAP Beeswarm Plot (Feature Influence Across Samples)")
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=300)
plt.show()

# ---- SHAP Summary ----
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
shap_ranking = pd.DataFrame({
    "Feature": cols,
    "Mean|SHAP|": mean_abs_shap
}).sort_values(by="Mean|SHAP|", ascending=False)
print("\n📈 SHAP Feature Ranking:\n", shap_ranking)

# ============================================================
# 2️⃣ PERMUTATION IMPORTANCE
# ============================================================
print("\n📊 Permutation Importance: Measuring performance drop per feature...")
perm = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)

perm_df = pd.DataFrame({
    "Feature": cols,
    "Importance": perm.importances_mean,
    "Std": perm.importances_std
}).sort_values(by="Importance", ascending=False)

print("\n📊 Permutation Importance Table:\n", perm_df)

# ---- Plot ----
plt.figure(figsize=(10,4))
sns.barplot(data=perm_df, x="Feature", y="Importance", palette="viridis")
plt.title("Permutation Feature Importance (Error Sensitivity)")
plt.ylabel("Mean Importance")
plt.tight_layout()
plt.savefig("permutation_importance.png", dpi=300)
plt.show()

# ============================================================
# 3️⃣ LIME EXPLANATION
# ============================================================
print("\n💡 LIME Explanation: Interpreting one example shape’s influence...")
lime_explainer = LimeTabularExplainer(
    X_train,
    feature_names=cols,
    mode='regression'
)

# Select one shape to explain
sample_idx = 5 if len(X_test) > 5 else 0
exp = lime_explainer.explain_instance(X_test[sample_idx], model.predict)

# ---- Display ----
print(f"\n🟢 LIME Explanation for Sample #{sample_idx}:")
for feat, weight in exp.as_list():
    print(f"  {feat}: {weight:.4f}")

fig = exp.as_pyplot_figure()
plt.title(f"LIME Local Explanation (Sample #{sample_idx})")
plt.tight_layout()
plt.savefig("lime_explanation.png", dpi=300)
plt.show()

# ============================================================
# 4️⃣ COMBINED SUMMARY
# ============================================================
print("\n✅ Explainability complete.")
print("Generated plots:")
print(" - shap_bar.png")
print(" - shap_beeswarm.png")
print(" - permutation_importance.png")
print(" - lime_explanation.png")

# ---- Save combined feature importance table ----
summary_df = pd.merge(shap_ranking, perm_df, on="Feature", how="outer").fillna(0)
summary_df.to_csv("feature_importance_summary.csv", index=False)
print("\n📁 Saved combined feature importances as feature_importance_summary.csv")
