import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import NotFittedError

# Optional plotting (matplotlib might not be installed in the current Python env)
try:
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    _HAS_MATPLOTLIB = False

# Optional NN training (tensorflow might not be installed)
try:
    import tensorflow as tf

    _HAS_TF = True
except ModuleNotFoundError:
    tf = None
    _HAS_TF = False


@dataclass
class Dataset:
    Re_train: list
    Re_val: list
    data_dir: str


def _read_snapshot(data_dir: str, Re: float) -> pd.DataFrame:
    fn = os.path.join(data_dir, f"steady_state_Re{int(Re)}.csv")
    if not os.path.exists(fn):
        raise FileNotFoundError(f"Missing snapshot: {fn}")
    return pd.read_csv(fn)


def _build_dp_masks_from_template(df_template: pd.DataFrame, q_up=0.2, q_down=0.8):
    # dp computed as average pressure difference between two axial locations.
    # We avoid the exact boundaries to reduce sensitivity to boundary artifacts.
    x = df_template["x"].values
    unique_x = np.unique(x)
    unique_x.sort()
    if len(unique_x) < 3:
        raise ValueError("Not enough unique x values to compute dp masks reliably.")

    i_up = int(q_up * (len(unique_x) - 1))
    i_down = int(q_down * (len(unique_x) - 1))
    x_up = unique_x[i_up]
    x_down = unique_x[i_down]

    dx = np.median(np.diff(unique_x)) if len(unique_x) > 1 else 1.0
    tol = max(1e-10, 0.5 * dx)

    mask_up = np.isclose(x, x_up, atol=tol)
    mask_down = np.isclose(x, x_down, atol=tol)

    if mask_up.sum() < 10 or mask_down.sum() < 10:
        raise ValueError(
            f"Mask too small: up={mask_up.sum()} down={mask_down.sum()}. "
            "Try different q_up/q_down."
        )

    return mask_up, mask_down, x_up, x_down, tol


def compute_dp_from_pressure_vector(p_vec: np.ndarray, mask_up: np.ndarray, mask_down: np.ndarray) -> float:
    p_up = float(np.mean(p_vec[mask_up]))
    p_down = float(np.mean(p_vec[mask_down]))
    return p_up - p_down


def compute_dp_from_snapshot(df: pd.DataFrame, mask_up: np.ndarray, mask_down: np.ndarray) -> float:
    p = df["p"].values.astype(float)
    return compute_dp_from_pressure_vector(p, mask_up=mask_up, mask_down=mask_down)


def pod_fit(P_snapshots: np.ndarray, n_modes: int):
    # P_snapshots: (n_Re, n_points)
    P_mean = P_snapshots.mean(axis=0, keepdims=True)
    P_centered = P_snapshots - P_mean
    svd = TruncatedSVD(n_components=n_modes)
    coeffs = svd.fit_transform(P_centered)
    modes = svd.components_  # (n_modes, n_points)
    return P_mean.squeeze(0), coeffs, modes


def pod_reconstruct(coeffs_pred: np.ndarray, modes: np.ndarray, P_mean: np.ndarray) -> np.ndarray:
    # coeffs_pred: (n_samples, n_modes) or (n_modes,)
    if coeffs_pred.ndim == 1:
        return coeffs_pred @ modes + P_mean
    return coeffs_pred @ modes + P_mean[None, :]


def train_nn_coeff_surrogate(Re_train: list, coeffs: np.ndarray, epochs: int = 200):
    if not _HAS_TF:
        raise ModuleNotFoundError("tensorflow is not available in this Python environment.")
    X = np.array(Re_train, dtype=float).reshape(-1, 1)
    y = coeffs.astype(float)

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    Xn = scaler_X.fit_transform(X)
    yn = scaler_Y.fit_transform(y)

    n_modes = y.shape[1]
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(n_modes),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    model.fit(
        Xn,
        yn,
        validation_split=0.2 if len(Re_train) >= 5 else 0.0,
        epochs=epochs,
        batch_size=min(4, len(Re_train)),
        verbose=0,
    )
    return model, scaler_X, scaler_Y


def predict_nn_coeff_surrogate(model, scaler_X, scaler_Y, Re_new: float) -> np.ndarray:
    X_new = np.array([[Re_new]], dtype=float)
    Xn = scaler_X.transform(X_new)
    pred_norm = model.predict(Xn, verbose=0)
    pred = scaler_Y.inverse_transform(pred_norm).flatten()
    return pred


def train_rf_coeff_surrogate(Re_train: list, coeffs: np.ndarray):
    X = np.array(Re_train, dtype=float).reshape(-1, 1)
    y = coeffs.astype(float)

    # Simple, interpretable coefficient surrogate: RandomForest per mode.
    # (We keep Re unscaled since trees are scale-invariant.)
    models = []
    for k in range(y.shape[1]):
        m = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=1,
        )
        m.fit(X, y[:, k])
        models.append(m)
    return models


def predict_rf_coeff_surrogate(models, Re_new: float) -> np.ndarray:
    X_new = np.array([[Re_new]], dtype=float)
    return np.array([m.predict(X_new)[0] for m in models], dtype=float)


def train_direct_dp_surrogate(Re_train: list, dp_train: list):
    X = np.array(Re_train, dtype=float).reshape(-1, 1)
    y = np.array(dp_train, dtype=float)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    return model


def main():
    data_dir = os.getcwd()

    Re_train = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    Re_val = [725]  # untrained validation Reynolds number(s)
    n_modes = 3

    print(f"Data dir: {data_dir}")
    print(f"Train Re: {Re_train}")
    print(f"Val Re: {Re_val}")

    # Load snapshots (pressure field vectors)
    # Use first training snapshot as template for dp mask locations.
    df0 = _read_snapshot(data_dir, Re_train[0])
    mask_up, mask_down, x_up, x_down, tol = _build_dp_masks_from_template(df0, q_up=0.2, q_down=0.8)
    print(f"dp masks: x_up={x_up:.6g}, x_down={x_down:.6g}, tol={tol:.3g}")

    P_snaps = []
    dp_train = []

    for Re in Re_train:
        df = _read_snapshot(data_dir, Re)
        p_vec = df["p"].values.astype(float)
        P_snaps.append(p_vec)
        dp_train.append(compute_dp_from_snapshot(df, mask_up, mask_down))

    P_snaps = np.stack(P_snaps, axis=0)  # (n_Re, n_points)
    dp_train = np.array(dp_train, dtype=float)

    # -------------------------
    # Model 1: POD(pressure) + Dense NN on POD coefficients
    # -------------------------
    P_mean, coeffs_train, modes = pod_fit(P_snaps, n_modes=n_modes)
    nn_model = None
    scaler_X = None
    scaler_Y = None
    if _HAS_TF:
        nn_model, scaler_X, scaler_Y = train_nn_coeff_surrogate(
            Re_train, coeffs_train, epochs=400
        )
    else:
        print("Skipping NN model: tensorflow not available.")

    # -------------------------
    # Model 2: POD(pressure) + RandomForest on POD coefficients
    # -------------------------
    rf_models = train_rf_coeff_surrogate(Re_train, coeffs_train)

    # -------------------------
    # Model 3: Direct regression Re -> dp
    # -------------------------
    direct_dp_model = train_direct_dp_surrogate(Re_train, dp_train)

    # Validation
    results = []
    for Re_new in Re_val:
        # True dp (if file exists)
        dp_true = None
        if os.path.exists(os.path.join(data_dir, f"steady_state_Re{int(Re_new)}.csv")):
            df_val = _read_snapshot(data_dir, Re_new)
            dp_true = compute_dp_from_snapshot(df_val, mask_up, mask_down)

        # Model 1 prediction
        dp_pred_nn = np.nan
        if _HAS_TF and nn_model is not None:
            coeffs_pred_nn = predict_nn_coeff_surrogate(nn_model, scaler_X, scaler_Y, Re_new)
            P_pred_nn = pod_reconstruct(coeffs_pred_nn, modes=modes, P_mean=P_mean)
            dp_pred_nn = compute_dp_from_pressure_vector(P_pred_nn, mask_up, mask_down)

        # Model 2 prediction
        coeffs_pred_rf = predict_rf_coeff_surrogate(rf_models, Re_new)
        P_pred_rf = pod_reconstruct(coeffs_pred_rf, modes=modes, P_mean=P_mean)
        dp_pred_rf = compute_dp_from_pressure_vector(P_pred_rf, mask_up, mask_down)

        # Model 3 prediction
        dp_pred_direct = float(direct_dp_model.predict(np.array([[Re_new]], dtype=float).reshape(-1, 1))[0])

        row = {
            "Re": Re_new,
            "dp_true": dp_true,
            "dp_pred_nn": dp_pred_nn,
            "dp_pred_rf": dp_pred_rf,
            "dp_pred_direct": dp_pred_direct,
        }
        results.append(row)

        print(f"\nRe={Re_new}:")
        print(f"  dp_true      = {dp_true}")
        print(f"  dp_pred_nn   = {dp_pred_nn:.6g}")
        print(f"  dp_pred_rf   = {dp_pred_rf:.6g}")
        print(f"  dp_pred_direct = {dp_pred_direct:.6g}")

        if dp_true is not None:
            for k in ["dp_pred_nn", "dp_pred_rf", "dp_pred_direct"]:
                err = abs(row[k] - dp_true)
                rel = err / abs(dp_true) if dp_true != 0 else np.nan
                print(f"  {k} abs_err={err:.6g}, rel_err={rel:.6g}")

    # Save summary
    out_csv = os.path.join(data_dir, "dp_model_results.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Simple plot (training dp vs Re + predicted points)
    if _HAS_MATPLOTLIB:
        plt.figure()
        plt.plot(Re_train, dp_train, "o-", label="True dp (train Re)")
        for row in results:
            Re_new = row["Re"]
            if np.isfinite(row["dp_pred_nn"]):
                plt.scatter([Re_new], [row["dp_pred_nn"]], marker="x", label="POD+NN")
            plt.scatter([Re_new], [row["dp_pred_rf"]], marker="^", label="POD+RF")
            plt.scatter([Re_new], [row["dp_pred_direct"]], marker="s", label="Direct")
        plt.xlabel("Re")
        plt.ylabel("dp (mean p_up - mean p_down)")
        plt.title("Pressure-drop prediction models")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(data_dir, "dp_models_plot.png")
        plt.savefig(plot_path, dpi=160)
        print(f"Saved plot: {plot_path}")
    else:
        print("matplotlib not available: skipping plot generation.")


if __name__ == "__main__":
    # TensorFlow can emit lots of logs; keep it quieter.
    if _HAS_TF:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        tf.get_logger().setLevel("ERROR")
    main()

