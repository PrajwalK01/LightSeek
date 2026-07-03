# Experiments Log

Track all training runs here.

## Format
| Run | Model | LR | Epochs | Val F1 | Notes |
|---|---|---|---|---|---|
| run_001 | CNN only | 1e-3 | 10 | - | Baseline |
| run_002 | CNN+Transformer | 1e-3 | 10 | - | Add Transformer |

## How to add a run
After each training run, add a row to this table with your results.
Save the model checkpoint as: `models/backbone/run_XXX.pth`
