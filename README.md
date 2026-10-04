# Federated Learning Across Three Simulated Hospitals

Can three hospitals train one good diagnostic model together without any of them sharing a single patient record?

This project implements **Federated Averaging (FedAvg) from scratch in NumPy** and compares it with hospitals training alone and with all data pooled in one place. It was motivated by my undergraduate research on EHR confidentiality at the University of Cape Coast Hospital, Ghana, where hospitals keep patient data siloed because they cannot share it safely.

**Public data only.** The project uses the Breast Cancer Wisconsin (Diagnostic) dataset bundled with scikit-learn. No real hospital records are used.

## Results

Averages over 20 random splits, ± one standard deviation, all scored on the same held-out test set.

| Approach | Accuracy | Sensitivity (malignant caught) | AUC |
| --- | --- | --- | --- |
| Alone: Hospital A (teaching, 255 patients) | 96.4% ± 1.4 | 94.5% ± 2.4 | 0.993 |
| Alone: Hospital B (regional, 128 patients) | 96.1% ± 1.5 | 91.4% ± 3.8 | 0.992 |
| Alone: Hospital C (district, 43 patients) | 94.2% ± 2.3 | 86.2% ± 5.4 | 0.989 |
| **Federated (FedAvg)** | **97.3% ± 1.2** | **95.2% ± 2.1** | **0.994** |
| Pooled data (no privacy) | 97.4% ± 1.0 | 94.9% ± 2.5 | 0.994 |

The federated model matched the pooled-data model while no hospital shared a record. The small district hospital's own model missed about 14 of every 100 malignant tumours; the federated model missed about 5.

![Sensitivity by approach](results/sensitivity_by_approach.png)

## How it works

1. The three hospitals differ in size and case mix (non-IID data), as real facilities do.
2. Feature normalisation is federated too: hospitals share only counts, sums and sums of squares.
3. Each round, every hospital trains the current logistic regression model on its own data for 5 epochs.
4. Hospitals send back only weights; the server averages them, weighted by hospital size.
5. 40 rounds in total. The local-only and pooled models get the same training budget.

## Run it

Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
python fedavg_hospitals.py
```

Takes a few seconds. Writes `results/results.csv` and two charts to `results/`. Use `--seeds 5` for a quicker run.

## Limitations

- One public dataset split three ways; real hospitals differ in equipment, formats and populations.
- Small, clean dataset with well-separated classes.
- FedAvg alone is not a privacy guarantee: model updates can leak information. Real systems add differential privacy or secure aggregation.
- All hospitals are assumed honest; a hospital sending poisoned updates could damage the shared model.

## Next steps

- Add differential privacy to measure the accuracy cost of formal privacy guarantees.
- Simulate a hospital that poisons its updates, and test robust aggregation defences.

## Author

[Richard Justice Doe](https://github.com/Richard-Justice-Doe)
