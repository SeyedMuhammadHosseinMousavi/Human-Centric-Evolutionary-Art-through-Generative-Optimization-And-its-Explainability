# ============================================================
# PSO-HUMAN MULTI-SHAPE XAI ANALYSIS (COMPATIBLE + FIXED)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# ------------------------ CONFIG ------------------------
MODEL_PATH = "pso_human_multishape_metrics_model.npz"
sns.set(style="whitegrid", font_scale=1.2)

# ------------------------ LOAD ------------------------
print(f"Loading model: {MODEL_PATH}")
data = np.load(MODEL_PATH, allow_pickle=True)

# --- Detect archive format ---
if "rectangles" in data:
    print("Detected Hill-Climbing archive format.")
    params = data["rectangles"]
    delta_error = data.get("delta_error", np.zeros(len(params)))
    errors = data["errors"]
    canvas_final = data.get("canvas_final", None)
    shape_types = ["rectangle"] * len(params)
    metrics = None
else:
    print("Detected PSO archive format.")
    params = data["params"]
    errors = data["errors"]
    delta_error = data["delta"]
    canvas_final = data["canvas_final"]
    shape_types = data["shapes"]
    metrics = data.get("metrics", None)

# --- Fix metrics if stored as ndarray ---
if isinstance(metrics, np.ndarray):
    try:
        metrics = metrics.item()
    except Exception:
        metrics = None

cols = ["x", "y", "w", "h", "angle", "R", "G", "B", "opacity"]
df = pd.DataFrame(params, columns=cols)
df["shape"] = shape_types

# ------------------------ PRINT SUMMARY ------------------------
print("\n🧠 Shape-by-Shape Analysis")
print("----------------------------------")
for i, (p, s) in enumerate(zip(params, shape_types)):
    print(f"[{i+1:03d}] Shape: {s:<10} | ΔError: {delta_error[i]:.6f} | "
          f"Pos: ({p[0]:.2f},{p[1]:.2f}) | Size: ({p[2]:.2f},{p[3]:.2f}) | "
          f"Angle: {p[4]:.2f} | Opacity: {p[8]:.2f} | "
          f"Color: ({p[5]:.2f},{p[6]:.2f},{p[7]:.2f})")

# ------------------------ FITNESS TREND ------------------------
plt.figure(figsize=(10,5))
plt.plot(errors, label="Error (MSE)", linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("Fitness Trend Over Time")
plt.legend()
plt.tight_layout()
plt.savefig("fitness_trend.png", dpi=300)
plt.show()

# ------------------------ ERROR CONTRIBUTION ------------------------
plt.figure(figsize=(10,4))
plt.bar(range(len(delta_error)), delta_error, color="skyblue")
plt.title("Error Reduction by Each Shape (ΔError)")
plt.xlabel("Shape Index")
plt.ylabel("Error Reduced (MSE)")
plt.tight_layout()
plt.savefig("error_contribution.png", dpi=300)
plt.show()

# ------------------------ PARAMETER VARIANCE ------------------------
variances = df.var(numeric_only=True).sort_values(ascending=False)
plt.figure(figsize=(10,4))
sns.barplot(x=variances.index, y=variances.values, palette="viridis")
plt.title("Parameter Variance (Proxy for Importance)")
plt.ylabel("Variance")
plt.tight_layout()
plt.savefig("parameter_variance.png", dpi=300)
plt.show()

# ------------------------ HEATMAP OF POSITIONS ------------------------
plt.figure(figsize=(5,5))
plt.hist2d(df["x"], df["y"], bins=32, range=[[0,1],[0,1]], cmap="magma")
plt.colorbar(label="Number of Shapes")
plt.title("Shape Placement Heatmap")
plt.xlabel("Normalized X")
plt.ylabel("Normalized Y")
plt.tight_layout()
plt.savefig("position_heatmap.png", dpi=300)
plt.show()

# ------------------------ COLOR & OPACITY TRENDS ------------------------
plt.figure(figsize=(10,4))
plt.plot(df["opacity"], label="Opacity", color="black", linewidth=1.5)
plt.plot(df["R"], label="Red", alpha=0.6)
plt.plot(df["G"], label="Green", alpha=0.6)
plt.plot(df["B"], label="Blue", alpha=0.6)
plt.legend()
plt.title("Opacity and Color Trends Over Time")
plt.xlabel("Shape Index")
plt.ylabel("Value (0–1)")
plt.tight_layout()
plt.savefig("opacity_color_trend.png", dpi=300)
plt.show()

# ------------------------ FINAL METRICS (OPTIONAL) ------------------------
if metrics:
    print("\n📊 Final Metrics:")
    for k, v in metrics.items():
        print(f"   {k:15s}: {v:.6f}")

print("\n✅ XAI analysis complete. Plots saved in current folder.")
