import pandas as pd

data = {
    "condition_id": ["C01","C02","C03","C04","C05","C06","C07","C08","C09","C10"],
    "real_world_rate":[0.10,0.22,0.20,0.24,0.18,0.25,0.21,0.15,0.14,0.19]
}

df = pd.DataFrame(data)

df.to_csv("namcs_benchmarks.csv", index=False)

print("Created namcs_benchmarks.csv")