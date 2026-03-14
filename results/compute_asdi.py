import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import seaborn as sns

df = pd.read_csv("results/llm_multimodel_scores.csv")

print("Rows loaded:", len(df))
print("Columns:", list(df.columns))

# Basic checks
if "condition" not in df.columns:
    raise ValueError("Missing column: condition")

if "mean_score" not in df.columns:
    raise ValueError("Missing column: mean_score")

df = df.dropna(subset=["condition", "mean_score"]).copy()
df["mean_score"] = pd.to_numeric(df["mean_score"], errors="coerce")
df = df.dropna(subset=["mean_score"]).copy()

print("\nMean score distribution:")
print(df["mean_score"].value_counts(dropna=False).sort_index())

print("\nCondition means:")
print(df.groupby("condition")["mean_score"].mean())

overall_mean = df["mean_score"].mean()
overall_std = df["mean_score"].std(ddof=0)

print("\nOverall mean:", overall_mean)
print("Overall std:", overall_std)

# Z-score handling
if overall_std == 0:
    df["z"] = 0.0
else:
    df["z"] = (df["mean_score"] - overall_mean) / overall_std

cond = df.groupby("condition", as_index=False)["mean_score"].mean()
cond = cond.sort_values("condition")
cond["amplification"] = cond["mean_score"] - overall_mean

ASDI_raw = np.mean(np.abs(cond["amplification"]))

cond_z = df.groupby("condition", as_index=False)["z"].mean()
cond_z = cond_z.sort_values("condition")
cond_z["amplification"] = cond_z["z"]

ASDI_z = np.mean(np.abs(cond_z["amplification"]))

# ANOVA
groups = [g["mean_score"].values for _, g in df.groupby("condition")]
if len(groups) > 1 and all(len(g) > 1 for g in groups):
    F, p = f_oneway(*groups)
else:
    F, p = np.nan, np.nan

print("\nANOVA p-value:", p)
print("ASDI raw:", ASDI_raw)
print("ASDI z:", ASDI_z)
# Effect size (Eta squared)
import numpy as np

grand_mean = df["mean_score"].mean()

ss_between = sum(
    len(group) * (group["mean_score"].mean() - grand_mean) ** 2
    for _, group in df.groupby("condition")
)

ss_total = sum(
    (df["mean_score"] - grand_mean) ** 2
)

eta_squared = ss_between / ss_total if ss_total != 0 else np.nan

print("Effect size (eta squared):", eta_squared)

# Save tables
cond.to_csv("results/final_paper_table_raw.csv", index=False)
cond_z.to_csv("results/final_paper_table_zscore.csv", index=False)

pd.DataFrame({
    "metric": ["overall_mean", "overall_std", "asdi_raw", "asdi_z", "anova_p_value"],
    "value": [overall_mean, overall_std, ASDI_raw, ASDI_z, p]
}).to_csv("results/asdi_metrics.csv", index=False)

# -------- Figure 1: BAR CHART --------
plt.figure(figsize=(10, 5))
plt.bar(cond["condition"].astype(str), cond["mean_score"])
plt.title("Follow-up Score by Condition")
plt.xlabel("Condition")
plt.ylabel("Mean Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/figure1_condition_scores.png", dpi=300)
plt.close()

# -------- Figure 2: HEATMAP --------
heatmap_df = cond[["condition", "mean_score"]].copy()
heatmap_df = heatmap_df.set_index("condition")

plt.figure(figsize=(4, 8))
sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlOrRd", cbar=True)
plt.title("Condition Mean Score Heatmap")
plt.tight_layout()
plt.savefig("results/figure2_condition_heatmap.png", dpi=300)
plt.close()

print("\nSaved:")
print("results/final_paper_table_raw.csv")
print("results/final_paper_table_zscore.csv")
print("results/asdi_metrics.csv")
print("results/figure1_condition_scores.png")
print("results/figure2_condition_heatmap.png")