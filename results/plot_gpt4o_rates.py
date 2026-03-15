import pandas as pd
import matplotlib.pyplot as plt

# read results file
df = pd.read_csv("results/gpt4o_binary_rates.csv")

plt.figure(figsize=(10,5))
plt.bar(df["condition"], df["gpt4o_binary"])

plt.ylim(0,1)
plt.ylabel("Screening Rate")
plt.xlabel("Condition")
plt.title("GPT-4o SDoH Screening Rates by Patient Condition")

plt.tight_layout()

plt.savefig("results/gpt4o_screening_rates.png", dpi=300)

print("Figure saved to results/gpt4o_screening_rates.png")
