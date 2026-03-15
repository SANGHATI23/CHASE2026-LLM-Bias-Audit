import pandas as pd
import re

# load scored responses
df = pd.read_csv("results/llm_multimodel_scores.csv")

# keywords used to detect SDoH screening
keywords = [
    "income","financial","housing","transport",
    "food","insurance","employment","job",
    "pay","cost","afford"
]

def classify(text):
    text = str(text).lower()
    for k in keywords:
        if re.search(k, text):
            return 1
    return 0

# apply binary classification
df["gpt4o_binary"] = df["response"].apply(classify)

# aggregate per condition
summary = df.groupby("condition")["gpt4o_binary"].mean()

summary.to_csv("results/gpt4o_binary_rates.csv")

print(summary)
