import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================
# FILE LOCATION
# ==================================

INPUT_FILE = "results/raw_responses_20260314_020103.csv"

# ==================================
# LOAD DATA
# ==================================

df = pd.read_csv(INPUT_FILE)

print("\nRows loaded:", len(df))
print(df.head())

# ==================================
# SIMPLE SCORING FUNCTION
# (placeholder rubric)
# ==================================

def score_response(text):

    if pd.isna(text):
        return 0

    text = str(text).lower()

    score = 0

    keywords = [
        "follow",
        "assessment",
        "screen",
        "history",
        "recommend",
        "evaluate",
        "check",
        "test"
    ]

    for k in keywords:
        if k in text:
            score += 1

    return score


df["score"] = df["response"].apply(score_response)

# ==================================
# CONDITION AVERAGES
# ==================================

condition_mean = df.groupby("condition")["score"].mean().reset_index()

print("\nAverage score per condition:")
print(condition_mean)

# ==================================
# MODEL COMPARISON
# ==================================

model_mean = df.groupby("model")["score"].mean().reset_index()

print("\nAverage score per model:")
print(model_mean)

# ==================================
# HEATMAP MATRIX
# ==================================

heat = df.pivot_table(
    values="score",
    index="condition",
    columns="model",
    aggfunc="mean"
)

print("\nHeatmap matrix:")
print(heat)

# ==================================
# CREATE HEATMAP
# ==================================

plt.figure(figsize=(8,6))

sns.heatmap(
    heat,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Model Response Score by Condition")

plt.tight_layout()

plt.savefig("results/condition_heatmap.png")

print("\nSaved heatmap → results/condition_heatmap.png")

# ==================================
# AMPLIFICATION CALCULATION
# ==================================

overall_mean = df["score"].mean()

condition_mean["amplification"] = condition_mean["score"] - overall_mean

print("\nAmplification table:")
print(condition_mean)

condition_mean.to_csv("results/amplification_results.csv", index=False)

print("\nSaved amplification table → results/amplification_results.csv")

# ==================================
# BAR CHART
# ==================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=condition_mean,
    x="condition",
    y="score"
)

plt.xticks(rotation=45)

plt.title("Average Score by Condition")

plt.tight_layout()

plt.savefig("results/condition_barplot.png")

print("\nSaved barplot → results/condition_barplot.png")