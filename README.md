# CHL-514 Final Project: ROM + ML Surrogates for Axisymmetric Pipe Flow

This project uses CFD (Basilisk) snapshots of axisymmetric pipe flow to build a **Reduced Order Model (ROM)** and **surrogate models** that can predict:

- the **velocity field** at an untrained Reynolds number `Re_new = 725`
- a scalar **pressure-drop metric** `dp` (computed from the CFD pressure field) at `Re_new = 725`

The key idea is:
1. Generate high-fidelity flow snapshots for multiple training Reynolds numbers.
2. Compress each snapshot using **POD/SVD** into a small number of modes.
3. Train regression models to predict the **POD coefficients** as a function of `Re`.
4. Reconstruct the full velocity/pressure-drop at an unseen `Re` and validate against CFD.

---

## What each file does

### CFD data generation
- `simulation.c`
  - Basilisk-based solver.
  - Runs axisymmetric pipe flow for a given `Re` and writes snapshot output to CSV.

- `generate_steady_states_wsl.sh`
  - Compiles and runs `simulation.c` with Basilisk `qcc` in WSL.
  - Produces files like:
    - `steady_state_Re100.csv`, `steady_state_Re500.csv`, ...

> Validation and ROM training depend on having the CSVs in this folder.

### ROM + surrogate training (velocity)
- `train_model.ipynb`
  - Main notebook for **velocity ROM**:
    - Load training velocity snapshots `steady_state_Re{Re}.csv`
    - Build POD basis (choose `n_modes`)
    - Train 3 alternative regressors in POD-coefficient space:
      - POD + Dense Neural Network (coefficients)
      - POD + Random Forest (coefficients)
      - POD + Gradient Boosting (coefficients)
    - Predict POD coefficients at `Re_new = 725`
    - Reconstruct the predicted velocity field `ux_pred`
    - Validate against CFD snapshot `steady_state_Re725.csv`
  - Also runs the `dp` modeling section at the end by calling:
    - `train_pressure_drop_models.py`

### Pressure-drop metric verification (visual + dp values)
- `verify_results.ipynb`
  - Computes and visualizes the scalar pressure-drop metric `dp` directly from the pressure snapshots.
  - Reports/plots results at the untrained validation point `Re = 725`.

### Pressure-drop surrogate training
- `train_pressure_drop_models.py`
  - Defines a consistent scalar `dp` from pressure snapshots.
  - Trains multiple surrogate approaches, outputs:
    - `dp_model_results.csv`
    - `dp_models_plot.png`

### Dependencies
- `requirements.txt`
  - Python packages needed for running the notebooks and training scripts.

### Generated datasets / outputs
- `steady_state_Re*.csv`
  - CFD snapshots used as training/validation data.
- `dp_model_results.csv`, `dp_models_plot.png`
  - Saved outputs for the pressure-drop surrogate evaluation.

---

## Theory (velocity ROM + ML surrogates)

### 1) Snapshot dataset
For each training Reynolds number `Re_train`, you have a CFD velocity snapshot:

- The notebook loads `steady_state_Re{Re_train}.csv`
- It extracts the velocity component (used as `ux`) into a vector form
- Stacking all training vectors forms a snapshot matrix `X`

So each training flow is represented as a point in a high-dimensional space.

### 2) POD/SVD: reduced basis
POD (Proper Orthogonal Decomposition) finds a small set of dominant spatial patterns that best represent your dataset in a least-squares sense.

In the notebook:
- Compute the mean snapshot: `X_mean`
- Center the data: `X_centered = X - X_mean`
- Run `TruncatedSVD` to compute:
  - `modes` (POD basis vectors, spatial patterns)
  - `coefficients` (how each training snapshot is expressed in the basis)

With `n_modes = 2`, the velocity field is approximated as:

`u(Re) ≈ X_mean + a1(Re)*phi1 + a2(Re)*phi2`

where:
- `phi1, phi2` are the POD modes
- `a1(Re), a2(Re)` are POD coefficients

This is the ROM compression step.

### 3) Surrogate regression in latent space
Now the problem becomes:

`Re  ->  [a1(Re), a2(Re)]`

Instead of predicting thousands of velocity values directly, you predict only the 2 POD coefficients.

To satisfy the “multiple models, not copy-paste” requirement, you compare 3 regressors:
- **Dense NN**: learns nonlinear mapping from `Re` to the coefficient vector
- **Random Forest**: ensemble of decision trees; handles nonlinearities differently
- **Gradient Boosting**: sequential tree ensemble; different inductive bias again

All three models are trained on the same latent targets (`coefficients`).

### 4) Reconstruction at the untrained Reynolds number
At `Re_new = 725`:
- Predict POD coefficients using each regressor
- Reconstruct the full velocity field using the POD basis:

`ux_pred = a_pred @ modes + X_mean`

### 5) Validation
Validation compares predicted velocity to CFD truth from `steady_state_Re725.csv`:
- Visual comparison via contour plots of `ux_pred` vs `ux_true`
- Error via:
  - absolute error field `|ux_true - ux_pred|`
  - global L2-type norm (`np.linalg.norm`)
- Additional interpretability via a profile/slice check (using a radial slice, e.g. at `R = 0.5`)

---

## Theory (pressure-drop modeling and `dp`)

### What is `dp` in this project?
`dp` is computed from the CFD pressure snapshots by taking a mean pressure difference between two axial locations:

`dp = mean(p_up) - mean(p_down)`

The pressure-drop script uses interior axial locations (not boundaries) to reduce sensitivity to boundary artifacts.

### Surrogate modeling for `dp`
The pressure-drop training script trains multiple approaches, including:
- POD on the pressure field + ML surrogate on POD coefficients
- direct ML regression: `Re -> dp`

Results are saved in `dp_model_results.csv` and visualized in `dp_models_plot.png`.

---

## Run sequence (what you should execute in order)

### A) Generate CFD CSV snapshots (must include Re=725)
1. Edit `generate_steady_states_wsl.sh`:
   - Add `725` to `RE_LIST` so the dataset includes `steady_state_Re725.csv`
2. In WSL, run:
   - `bash generate_steady_states_wsl.sh`
3. Verify that these files exist in the project folder:
   - `steady_state_Re{100,500,1000,1500,2000,2500,3000,3500,4000,4500,5000}.csv`
   - `steady_state_Re725.csv`

### B) Train and validate velocity ROM
1. Open `train_model.ipynb`
2. Run cells top-to-bottom (this trains POD + ML surrogates and reconstructs at `Re_new = 725`)

### C) Validate/visualize pressure-drop
1. Open `verify_results.ipynb`
2. Run cells top-to-bottom to compute/plot `dp` vs `Re` and report the untrained point at `Re = 725`

---

## Expected outputs

After running the notebook(s), you should produce/see:
- velocity ROM validation plots and L2 error numbers at `Re=725`
- `dp_model_results.csv`
- `dp_models_plot.png`

---