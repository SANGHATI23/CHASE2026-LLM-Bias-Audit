# CHASE2026 LLM Bias Audit

This repository contains the experimental pipeline used to evaluate bias amplification in LLM follow-up recommendations.

## Experiment Design

- 10 prompt conditions
- 15 runs per condition
- 150 total responses

Models were evaluated using three rubric scorers.

## Metrics

- Mean follow-up score
- ASDI (Amplification Score for Disparity Index)
- ANOVA
- Effect size

## Key Results

Mean score: 3.90  
ASDI: 0.073  
ANOVA p-value: 0.87

Indicating **low amplification across SDOH prompt conditions**.

## Figures

- `figure1_condition_scores.png`
- `figure2_condition_heatmap.png`

## Files

| Script | Purpose |
|------|------|
| run_audit.py | Generate responses |
| llm_rubric_score.py | Score outputs |
| compute_asdi.py | Bias metric |
| final_metrics.py | Paper statistics |
