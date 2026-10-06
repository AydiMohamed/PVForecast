# Short-Term PV Power Forecasting: Tree Ensembles vs. a CNN-LSTM Hybrid

Code for the paper **"Short-Term Photovoltaic Power Forecasting: A Dual-Path Benchmark of Tree Ensembles and a CNN-LSTM Hybrid"** (Aydi, Aoudni, Ktari, ENIS Sfax).

The study forecasts the total AC power of a 4.74 MW utility-scale PV plant (PVDAQ Site 9068, Kersey, Colorado) **1 h and 2 h ahead** using plant measurements and historical weather only (no sky or satellite imagery). It compares:

| Model | Input |
|---|---|
| Persistence | latest AC power |
| Linear Regression, Random Forest, XGBoost | 30 current features + lagged values of 4 signals (130 inputs) |
| XGBoost (no lag) | the 30 current features only |
| LSTM, CNN+LSTM (Conv1D), CNN(2D)+LSTM (ablation) | 24-step window of the 30 features |

All models use the same chronological split and test period.

## Repository contents

```
.
├── PV_Forecasting_Cleaned_v7_review.ipynb   # full pipeline: preprocessing, models, evaluation
├── requirements.txt                         # package versions
├── LICENSE
└── README.md
```


## Data

The data are public and are **not included** in this repository.

1. **PVDAQ Site 9068** (NREL): download from the PVDAQ public datasets, DOI [10.25984/1846021](https://doi.org/10.25984/1846021). The notebook expects these files in one folder:
   - `9068_ac_power_data.csv`, `9068_dc_combiner_data.csv`, `9068_environment_data.csv`, `9068_irradiance_data.csv`, `9068_tracker_data.csv` (training and validation period)
   - the matching `..._20240101_20250430.csv` files (test period, January 2024 to April 2025)
2. **Open-Meteo Historical Weather API**: hourly data for the plant location, saved as `9068_open_meteo_data.csv`. Variables used: apparent temperature, day flag, direct / shortwave / diffuse radiation, direct normal irradiance, cloud cover (precipitation is dropped).

Set the path to your data folder in the first configuration cell of the notebook (variable `data_folder`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Main packages: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `torch`, `matplotlib`, `tqdm`.

A GPU is optional. XGBoost and the PyTorch models use CUDA when it is available; Random Forest runs on CPU.

## How to run

Run the notebook from top to bottom:

1. **Sections 1-3:** configuration, cleaning functions, and the dataset builder (5-minute grid, physical-range masking, interpolation, cyclical time features, targets at +1 h and +2 h).
2. **Sections 4-6:** lagged features for the tree and linear models, scaling (fit on the training period only), and lazy sliding windows for the neural models.
3. **Section 8 onward:** Linear Regression, Random Forest and XGBoost (randomized search), the daytime-only experiment, LSTM, CNN+LSTM, the Conv2D ablation, and persistence.
4. **Section 14:** the final comparison table.

## Experimental setup

- **Resolution and horizons:** 5 minutes; targets are the total AC power (sum of the two inverters) 12 and 24 steps ahead.
- **Split (chronological):** training through mid-2022 (523,342 samples), validation through the end of 2023 (131,881), test January 2024 to April 2025 (139,657).
- **Lag features:** total AC power, DC power of inverter 1, and the two pyranometers, each at the current step and the previous 24 steps (100 lag columns).
- **Tree model tuning:** randomized search, 10 candidates, on 100,000 training and 100,000 validation rows, scored by RMSE on the scaled target. Search spaces and selected values are in Table III of the paper.
- **Neural models (not tuned):** 2-layer LSTM, 64 hidden units, dropout 0.2; Adam (lr 1e-3, weight decay 1e-4); MSE loss; batch size 128; gradient clipping at 1.0; ReduceLROnPlateau (factor 0.5, patience 5); up to 50 epochs with early stopping (patience 10). CNN+LSTM adds two Conv1D layers (64 filters, kernel 3).
- **Seeds:** 42, 100 and 200. Reported results are mean ± sample standard deviation over the three seeds.

## Results (test set, RMSE in kW, mean ± std over 3 seeds)

| Model | 1 h | 2 h |
|---|---|---|
| Persistence | 691.7 | 1000.3 |
| Linear Regression | 562.2 | 706.6 |
| Random Forest | 473.9 ± 0.1 | 564.2 ± 0.9 |
| XGBoost | 470.1 ± 0.4 | 561.1 ± 1.1 |
| XGBoost (no lag) | 491.7 ± 12.0 | 585.3 ± 20.2 |
| LSTM | 611.5 ± 28.9 | 688.8 ± 28.2 |
| CNN+LSTM | 602.7 ± 45.6 | 680.1 ± 46.0 |

See the paper for skill scores, the per-seed values, and the discussion.

## Reproducibility notes

- Seeds are set with `torch.manual_seed` for the neural models and `random_state` for the tree models.
- Training resumes from a checkpoint if one exists. Delete the `best_*_model.pth` files to retrain from scratch.
- GPU execution can introduce small numerical differences between machines.

## Citation

```bibtex
@inproceedings{aydi2026pv,
  title     = {Short-Term Photovoltaic Power Forecasting: A Dual-Path Benchmark of Tree Ensembles and a CNN--LSTM Hybrid},
  author    = {Aydi, Mohamed and Aoudni, Yassine and Ktari, Jalel},
  year      = {2026}
}
```


## License

Code released under the [MIT License](LICENSE) . The PVDAQ and Open-Meteo data are subject to their own terms; see the links above.

## Contact

Mohamed Aydi, mohamed.aydi@enis.tn
