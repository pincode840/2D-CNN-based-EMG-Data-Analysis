# 2D CNN Based EMG Data Analysis

This repository contains Jupyter notebook experiments for classifying hand
gesture EMG data with image-shaped neural-network inputs. The main idea is to
reshape multi-channel EMG windows into a 2D representation and train CNN-based
models.

## Dataset

The `hand_data/` folder contains four CSV files:

| File | Rows | Channels | Meaning |
| --- | ---: | ---: | --- |
| `hand_data/0.csv` | 2910 | 8 | Gesture class 0 |
| `hand_data/1.csv` | 2903 | 8 | Gesture class 1 |
| `hand_data/2.csv` | 2943 | 8 | Gesture class 2 |
| `hand_data/3.csv` | 2922 | 8 | Gesture class 3 |

Each row is an 8-channel EMG sample. The notebooks build labeled training data
from these class-specific files.

## Notebook Summary

| Notebook | What It Does |
| --- | --- |
| `final.ipynb` | Main 2D CNN experiment using TensorFlow/Keras, scaling and train/test split |
| `hand1.ipynb` | Earlier LSTM/CNN style experiment for the same hand data |
| `hand3.ipynb` | CNN experiment with evaluation outputs such as confusion matrix |

## Observed Training Logs

The notebooks include saved output logs. Example validation accuracy values
visible in the notebook outputs are around the mid-to-high 90% range for the
later CNN experiments. Treat these as notebook-run records, not as a formally
reproduced benchmark.

## How To Run

```bash
pip install -r requirements.txt
jupyter notebook final.ipynb
```

## Notes

- The current structure is notebook-first and good for analysis history.
- If this becomes a paper or portfolio project, the next step is to move data
  loading, preprocessing, model definition and evaluation into Python modules.
- Add train/test split details and a fixed random seed before comparing models.
