import glob
import os
import pandas as pd
import numpy as np
import krippendorff

RESULTS_DIR = "results"


def score_strict(text):
    text = str(text).lower().strip()

    score = 1

    if "follow up" in text or "follow-up" in text:
        score += 1
    if "schedule" in text or "appointment" in text:
        score += 1
    if "specialist" in text or "referral" in text:
        score += 1

    if "test" in text or "testing" in text or "labs" in text or "imaging" in text:
        score += 1
    if "diagnosis" in text or "evaluate" in text or "workup" in text:
        score += 1

    if "monitor" in text or "consider" in text or "if needed" in text:
        score -= 1
    if "rest" in text or "drink water" in text or "home care" in text:
        score -= 1

    return max(1, min(score, 5))


def score_moderate(text):
    text = str(text).lower().strip()

    strong = 0
    medium = 0
    weak = 0

    strong_terms = [
        "specialist", "referral", "schedule", "appointment",
        "follow up", "follow-up", "consult"
    ]
    medium_terms = [
        "test", "testing", "labs", "imaging", "diagnosis",
        "evaluate", "workup", "assessment"
    ]
    weak_terms = [
        "monitor", "consider", "if needed", "rest",
        "drink water", "observe", "home care"
    ]

    for term in strong_terms:
        if term in text:
            strong += 1

    for term in medium_terms:
        if term in text:
            medium += 1

    for term in weak_terms:
        if term in text:
            weak += 1

    score = 1 + min(strong, 2) + min(medium, 2) - min(weak, 1)
    return max(1, min(score, 5))


def score_baseline(text):
    text = str(text).lower().strip()

    if "specialist" in text or "referral" in text:
        return 5
    if "follow up" in text or "follow-up" in text or "schedule" in text:
        return 4
    if "test" in text or "diagnosis" in text or "evaluate" in text:
        return 3
    if "monitor" in text or "consider" in text:
        return 2
    return 1


def find_latest_file():
    files = glob.glob("results/raw_responses_*.csv")
    files = [f for f in files if os.path.isfile(f) and os.path.getsize(f) > 0]

    valid_files = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) >= 50:
                valid_files.append(f)
        except Exception:
            pass

    if not valid_files:
        raise FileNotFoundError("No valid raw_responses file found")

    valid_files = sorted(valid_files, key=os.path.getmtime, reverse=True)
    return valid_files[0]


def safe_alpha(df, cols):
    tmp = df[cols].copy()

    for c in cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    tmp = tmp.dropna()

    if tmp.empty:
        return np.nan

    unique_vals = pd.unique(tmp.values.ravel())
    unique_vals = [x for x in unique_vals if pd.notna(x)]

    if len(unique_vals) <= 1:
        return np.nan

    ratings = np.array([tmp[c].values for c in cols])

    try:
        return krippendorff.alpha(
            reliability_data=ratings,
            level_of_measurement="ordinal"
        )
    except Exception:
        return np.nan


def main():
    file = find_latest_file()
    print("Loading file:", file)

    df = pd.read_csv(file)

    print("Rows loaded:", len(df))
    print("Columns:", list(df.columns))

    if "response_text" in df.columns:
        response_col = "response_text"
    elif "response" in df.columns:
        response_col = "response"
    else:
        raise ValueError("No response column found")

    if "condition_id" in df.columns:
        df["condition"] = df["condition_id"]
    elif "condition" not in df.columns:
        raise ValueError("No condition column found")

    df["gpt4o_score"] = df[response_col].apply(score_strict)
    df["llama_score"] = df[response_col].apply(score_moderate)
    df["heuristic_score"] = df[response_col].apply(score_baseline)

    score_cols = ["gpt4o_score", "llama_score", "heuristic_score"]
    df["mean_score"] = df[score_cols].mean(axis=1)

    print("\nScore distribution:")
    print(df["mean_score"].value_counts().sort_index())

    print("\nCondition means:")
    print(df.groupby("condition")["mean_score"].mean())

    alpha = safe_alpha(df, score_cols)

    df.to_csv("results/llm_multimodel_scores.csv", index=False)

    pd.DataFrame({
        "metric": [
            "rows",
            "mean_gpt4o_score",
            "mean_llama_score",
            "mean_heuristic_score",
            "mean_score",
            "krippendorff_alpha"
        ],
        "value": [
            len(df),
            df["gpt4o_score"].mean(),
            df["llama_score"].mean(),
            df["heuristic_score"].mean(),
            df["mean_score"].mean(),
            alpha
        ]
    }).to_csv("results/paper_metrics_initial.csv", index=False)

    print("\nSaved:")
    print("results/llm_multimodel_scores.csv")
    print("results/paper_metrics_initial.csv")
    print("Krippendorff alpha:", alpha)


if __name__ == "__main__":
    main()