"""Random-search hyper-parameter optimisation (synopsis 5.5). Objective = validation F1."""
import random, time
import numpy as np
from sklearn.metrics import f1_score
import tensorflow as tf
import config as C
from model import build_cnn


def sample_config(rng):
    return {k: rng.choice(v) for k, v in C.SEARCH_SPACE.items()}


def random_search(data, cw, n_trials=C.N_TRIALS, epochs=C.TUNE_EPOCHS, seed=C.SEED):
    rng = random.Random(seed)
    results, seen = [], set()
    for t in range(n_trials):
        cfg = sample_config(rng)
        key = tuple(sorted(cfg.items()))
        if key in seen:
            continue
        seen.add(key)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(C.SEED)
        model = build_cnn(**cfg)
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        t0 = time.time()
        model.fit(data["X_train"], data["y_train"], validation_data=(data["X_val"], data["y_val"]),
                  epochs=epochs, batch_size=cfg["batch_size"], class_weight=cw,
                  callbacks=[es], verbose=0)
        prob = model.predict(data["X_val"], verbose=0).ravel()
        f1 = f1_score(data["y_val"], (prob >= C.THRESHOLD).astype(int))
        loss = model.evaluate(data["X_val"], data["y_val"], verbose=0)[0]
        results.append(dict(trial=t + 1, val_f1=float(f1), val_loss=float(loss), **cfg))
        print(f"  trial {t+1:>2}/{n_trials}  val_F1={f1:.4f}  val_loss={loss:.4f}  "
              f"{cfg}  ({time.time()-t0:.0f}s)")
    results.sort(key=lambda r: (-r["val_f1"], r["val_loss"]))
    best = {k: results[0][k] for k in C.SEARCH_SPACE}
    return best, results
