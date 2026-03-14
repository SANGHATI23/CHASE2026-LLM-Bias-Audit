import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("results/amplification_results.csv")

plt.figure(figsize=(6,4))

sns.heatmap(
    df.pivot_table(values="amplification", index="condition"),
    annot=True,
    cmap="coolwarm",
    center=0
)

plt.title("Amplification by Condition")

plt.tight_layout()

plt.savefig("results/amplification_heatmap.png")