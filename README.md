# CHASE2026 LLM Bias Audit

This repository contains the experimental pipeline for the paper:
**"Can Generative AI Mitigate or Amplify Screening Disparities? 
A Fairness Audit and Gap-Detection Framework for Social Determinants of Health Screening"**
Submitted to CHASE 2026 Workshop on Generative AI for Smart Health and Biomedical Informatics.

## Experiment Design
- 50 vignettes × 2 models × 3 runs = 300 total responses
- Binary SDoH screening classification (keyword rule-based)
- 10 demographic conditions (race, income, disability, community type)

## Models Evaluated
- Llama 3 70B (temperature=0)
- GPT-4o (temperature=0)

## Key Results
| Model | DPD | ASDI | Mean Screening Rate |
|-------|-----|------|---------------------|
| Llama 3 70B | 93 pp | 0.395 | 80.0% |
| GPT-4o | 73 pp | 0.312 | 76.7% |

C05 (White/Physical Disability/Suburban) shows consistent suppression across both models (7% and 40% vs. 55% real-world).

## Files
| Script | Purpose |
|--------|---------|
| `run_audit.py` | Generate LLM responses |
| `analyze_results.py` | Binary keyword classification |
| `make_heatmap.py` | Generate figures |
| `namcs_benchmark.py` | Real-world benchmark comparison |
| `vignettes.csv` | 50 demographic vignettes |

## Results
See `/results` folder for figures and CSVs.
