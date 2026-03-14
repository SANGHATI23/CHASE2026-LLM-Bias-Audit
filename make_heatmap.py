import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results/table1.csv")

# Labels for conditions
label_map = {
    "C01": "C01 White / High Income / Urban",
    "C02": "C02 Black / Low Income / Urban",
    "C03": "C03 Hispanic / Low Income / Urban",
    "C04": "C04 AIAN / Low Income / Rural",
    "C05": "C05 White / Physical Disability / Suburban",
    "C06": "C06 Black / Physical Disability / Urban",
    "C07": "C07 Hispanic / Cognitive Disability / Suburban",
    "C08": "C08 Asian / Mid Income / Urban",
    "C09": "C09 MENA / Mid Income / Urban",
    "C10": "C10 White / Cognitive Disability / Rural"
}

df["condition_label"] = df["condition_id"].map(label_map)

# Sort conditions
df = df.sort_values("condition_id")

# Prepare heatmap data
heatmap_data = df[["sdoh_any"]].values

fig, ax = plt.subplots(figsize=(6,8))

im = ax.imshow(heatmap_data, aspect="auto")

# Axis labels
ax.set_xticks([0])
ax.set_xticklabels(["Llama"])

ax.set_yticks(range(len(df)))
ax.set_yticklabels(df["condition_label"])

# Show values inside cells
for i, val in enumerate(df["sdoh_any"]):
    ax.text(0, i, f"{val:.2f}", ha="center", va="center", color="black")

ax.set_title("SDoH Screening Rate by Demographic Condition")
ax.set_xlabel("Model")
ax.set_ylabel("Condition")

cbar = plt.colorbar(im)
cbar.set_label("Screening Rate")

plt.tight_layout()

plt.savefig("results/fairness_heatmap.png", dpi=300)

print("Heatmap saved to results/fairness_heatmap.png")