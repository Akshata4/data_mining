# SEMMA Titanic — End-to-End Bundle

Sample -> Explore -> Modify -> Model -> Assess for Kaggle Titanic.

- Group-aware 70/15/15 split by Ticket_norm
- Pipeline-in-CV (leakage-safe)
- Calibrator chosen by Brier on VAL (sigmoid vs isotonic)
- Threshold sensitivity (FN:FP grid) + bootstrap CIs
- Fairness gaps + basic drift checks

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optionally place Kaggle `train.csv` in ./data/raw/
make sample explore modify model assess
```
