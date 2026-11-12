"""Full pipeline: data -> baseline CNN -> optimisation -> optimised CNN -> test evaluation -> Grad-CAM.

Usage:
    python train.py                # full run
    python train.py --quick        # fast demo (fewer trials / epochs)
"""
import os, json, argparse, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import tensorflow as tf
import config as C
import data_utils as D
from model import build_cnn
from tune import random_search
import evaluate as E
import explain as xai


def fit_final(params, data, cw, name):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(C.SEED)
    model = build_cnn(**params)
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE,
                                            restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)]
    h = model.fit(data["X_train"], data["y_train"], validation_data=(data["X_val"], data["y_val"]),
                  epochs=C.MAX_EPOCHS, batch_size=params["batch_size"], class_weight=cw,
                  callbacks=cbs, verbose=2)
    E.plot_history(h.history, C.OUT_DIR / f"history_{name}.png", name)
    return model


def main(quick=False):
    t0 = time.time()
    if quick:
        C.MAX_EPOCHS, C.N_TRIALS, C.TUNE_EPOCHS = 12, 4, 8
    # 1) data ------------------------------------------------------------
    print("\n=== 1. Loading & validating data ===")
    X, y, source = D.load_dataset()
    X, y, report = D.validate(X, y)
    print(source); print(json.dumps(report, indent=2))
    data, stats = D.split_and_normalise(X, y)
    cw = D.class_weights(data["y_train"])
    print(f"train/val/test = {len(data['y_train'])}/{len(data['y_val'])}/{len(data['y_test'])}  class_weight={cw}")
    json.dump(stats, open(C.OUT_DIR / "norm_stats.json", "w"))

    # 2) baseline --------------------------------------------------------
    print("\n=== 2. Baseline CNN ===")
    base = fit_final(C.BASELINE_PARAMS, data, cw, "baseline")
    p_base = base.predict(data["X_test"], verbose=0).ravel()
    m_base = E.compute_metrics(data["y_test"], p_base)

    # 3) optimisation (validation data only) -----------------------------
    print(f"\n=== 3. Hyper-parameter optimisation ({C.N_TRIALS} random-search trials) ===")
    best, trials = random_search(data, cw, C.N_TRIALS, C.TUNE_EPOCHS)
    pd.DataFrame(trials).to_csv(C.OUT_DIR / "tuning_trials.csv", index=False)
    json.dump(best, open(C.OUT_DIR / "best_params.json", "w"), indent=2)
    print("Best configuration:", best)

    # 4) final optimised model ------------------------------------------
    print("\n=== 4. Training optimised CNN ===")
    opt = fit_final(best, data, cw, "optimized")
    opt.save(C.OUT_DIR / "heart_attack_cnn.keras")
    p_opt = opt.predict(data["X_test"], verbose=0).ravel()
    m_opt = E.compute_metrics(data["y_test"], p_opt)

    # 5) test evaluation -------------------------------------------------
    print("\n=== 5. Test-set evaluation ===")
    cmp = pd.DataFrame({"baseline": m_base, "optimized": m_opt}).T
    print(cmp.round(4).to_string())
    cmp.to_csv(C.OUT_DIR / "metrics_comparison.csv")
    E.plot_confusion(m_base, C.OUT_DIR / "confusion_baseline.png", "Baseline CNN")
    E.plot_confusion(m_opt, C.OUT_DIR / "confusion_optimized.png", "Optimized CNN")
    E.plot_roc(data["y_test"], {"Baseline": p_base, "Optimized": p_opt}, C.OUT_DIR / "roc_curve.png")

    # 6) explainability + error analysis --------------------------------
    print("\n=== 6. Grad-CAM & error analysis ===")
    pred = (p_opt >= C.THRESHOLD).astype(int); yt = data["y_test"]
    cases = {"true_positive": (yt == 1) & (pred == 1), "true_negative": (yt == 0) & (pred == 0),
             "false_positive": (yt == 0) & (pred == 1), "false_negative": (yt == 1) & (pred == 0)}
    for name, mask in cases.items():
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        i = idx[0]
        cam, prob = xai.grad_cam_1d(opt, data["X_test"][i:i + 1])
        xai.plot_gradcam(data["X_test"][i, :, 0], cam, prob, C.OUT_DIR / f"gradcam_{name}.png", name + ":")

    json.dump({"data_source": source, "data_report": report, "baseline": m_base,
               "optimized": m_opt, "best_params": best}, open(C.OUT_DIR / "results.json", "w"),
              indent=2, default=float)
    print(f"\nDone in {time.time()-t0:.0f}s. Outputs saved in: {C.OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
