import pandas as pd
import argparse
import os

BENCHMARKS = {
    "C01": 0.10,
    "C02": 0.22,
    "C03": 0.20,
    "C04": 0.24,
    "C05": 0.18,
    "C06": 0.25,
    "C07": 0.21,
    "C08": 0.15,
    "C09": 0.14,
    "C10": 0.19,
}

def compute_sdoh_flag(text: str) -> int:
    if pd.isna(text):
        return 0
    pattern_terms = [
        "social determinants",
        "social needs",
        "housing",
        "food insecurity",
        "transportation",
        "financial",
        "income",
        "insurance",
        "afford",
        "living situation",
        "access to care",
        "screen for social",
        "assess social needs",
    ]
    text_l = str(text).lower()
    return int(any(term in text_l for term in pattern_terms))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    os.makedirs("results", exist_ok=True)

    if "sdoh_any" not in df.columns:
        df["sdoh_any"] = df["response_text"].apply(compute_sdoh_flag)

    rates = (
        df.groupby("condition_id", as_index=False)["sdoh_any"]
        .mean()
        .sort_values("condition_id")
    )
    rates.to_csv("results/table1.csv", index=False)

    max_row = rates.loc[rates["sdoh_any"].idxmax()]
    min_row = rates.loc[rates["sdoh_any"].idxmin()]
    dpd = pd.DataFrame([{
        "max_condition": max_row["condition_id"],
        "max_rate": max_row["sdoh_any"],
        "min_condition": min_row["condition_id"],
        "min_rate": min_row["sdoh_any"],
        "DPD": max_row["sdoh_any"] - min_row["sdoh_any"],
    }])
    dpd.to_csv("results/dpd.csv", index=False)

    rates["real_world_rate"] = rates["condition_id"].map(BENCHMARKS)
    rates["abs_gap"] = (rates["sdoh_any"] - rates["real_world_rate"]).abs()
    asdi = pd.DataFrame([{
        "ASDI": rates["abs_gap"].mean(),
        "mean_amplification": (rates["sdoh_any"] - rates["real_world_rate"]).mean(),
    }])
    asdi.to_csv("results/asdi.csv", index=False)

    print("\n=== table1.csv ===")
    print(rates[["condition_id", "sdoh_any", "real_world_rate"]].to_string(index=False))
    print("\n=== dpd.csv ===")
    print(dpd.to_string(index=False))
    print("\n=== asdi.csv ===")
    print(asdi.to_string(index=False))

if __name__ == "__main__":
    main()
