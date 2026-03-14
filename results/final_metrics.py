import pandas as pd

a = pd.read_csv("results/paper_metrics_initial.csv")
b = pd.read_csv("results/asdi_metrics.csv")

final = pd.concat([a,b])

final.to_csv("results/paper_metrics.csv", index=False)

print("Saved results/paper_metrics.csv")
print(final)