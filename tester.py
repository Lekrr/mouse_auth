import argparse
import os
import glob
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from config import TRUST_MODEL_PARAMS, ONE_CLASS_SVM_PARAMS, TTIME_VALUE, MIN_N_ACTION
from src.pipeline.model import TrustModel

from window_extraction import WINDOW_EXTRACTORS, UserFeatures

DATASET_DIT = "./dataset"
TRAIN_RATIO = 0.7
RANDOM_SEED = 42
DEFAULT_WINDOW_MODE = "time"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger(__name__)


def _cache_path(window_mode: str) -> Path:
    return Path(f"./features_cache_{window_mode}.pkl")


def load_user_features(user_dir: str,
                       min_n_action: int = MIN_N_ACTION,
                       ttime_value: float = TTIME_VALUE,
                       window_mode: str = DEFAULT_WINDOW_MODE,
                       extraction_function=None,
                       ) -> UserFeatures:
    windows: UserFeatures = []

    pattern = os.path.join(user_dir, "**", "MOUSE.csv")
    mouse_files = sorted(glob.glob(pattern, recursive=True))
    if not mouse_files:
        log.warning("No MOUSE.csv found under %s", user_dir)
        return windows

    if window_mode not in WINDOW_EXTRACTORS:
        raise ValueError(
            f"Unknown window_mode {window_mode!r}; "
            f"valid values: {sorted(WINDOW_EXTRACTORS)}"
        )
    extractor_fn = WINDOW_EXTRACTORS[window_mode]

    for mouse_csv in mouse_files:
        session_windows = extractor_fn(
            mouse_csv, min_n_action,
            extraction_function=extraction_function,
        )
        if session_windows:
            windows.extend(session_windows)

    windows.sort(key=lambda w: w[0])
    return windows


def load_all_users(dataset_dir: str,
                   window_mode: str = DEFAULT_WINDOW_MODE,
                   extraction_function=None,
                   ) -> Dict[str, UserFeatures]:
    users: Dict[str, UserFeatures] = {}
    for user_id in sorted(os.listdir(dataset_dir)):
        user_dir = os.path.join(dataset_dir, user_id)
        if not os.path.isdir(user_dir):
            continue

        windows = load_user_features(
            user_dir,
            window_mode=window_mode,
            extraction_function=extraction_function,
        )
        if not windows:
            log.warning("User %s has no usable windows", user_id)
            continue
        users[user_id] = windows
        log.info("User %-12s : %d windows", user_id, len(windows))

    return users


def _save_features(users: Dict[str, UserFeatures], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(users, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cache_path)


def _load_features(cache_path: Path) -> Dict[str, UserFeatures]:
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def get_features(dataset_dir: str = DATASET_DIT,
                 force_recompute: bool = False,
                 window_mode: str = DEFAULT_WINDOW_MODE,
                 extraction_function=None,
                 ) -> Dict[str, UserFeatures]:
    if window_mode not in WINDOW_EXTRACTORS:
        raise ValueError(
            f"Unknown window_mode {window_mode!r}; "
            f"valid values: {sorted(WINDOW_EXTRACTORS)}"
        )
    cache_path = _cache_path(window_mode)

    if not force_recompute and cache_path.exists():
        try:
            users = _load_features(cache_path)
            log.info("Loaded cached features from %s (%d users, mode=%s)",
                     cache_path, len(users), window_mode)
            return users
        except Exception as exc:                                # noqa: BLE001
            log.warning("Could not read cache %s: %s — rebuilding",
                        cache_path, exc)

    log.info("Extracting features (window_mode=%s)…", window_mode)
    users = load_all_users(dataset_dir,
                           window_mode=window_mode,
                           extraction_function=extraction_function)
    if not users:
        raise RuntimeError("No users with usable data found.")

    _save_features(users, cache_path)
    log.info("Saved features to %s", cache_path)
    return users


def split_train_test(windows: UserFeatures,
                     train_ratio: float = TRAIN_RATIO,
                     seed: int = RANDOM_SEED,
                     ) -> Tuple[UserFeatures, UserFeatures]:
    n = len(windows)
    if n < 2:
        return [], []

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = max(1, int(round(n * train_ratio)))
    n_train = min(n_train, n - 1)

    train = [windows[i] for i in idx[:n_train]]
    test = [windows[i] for i in idx[n_train:]]

    train.sort(key=lambda w: w[0])
    test.sort(key=lambda w: w[0])
    return train, test


def evaluate_user(user_id: str,
                  train: UserFeatures,
                  test_own: UserFeatures,
                  test_others: Dict[str, UserFeatures],
                  ) -> Dict[str, float]:
    model = TrustModel(
        user_id=user_id,
        **TRUST_MODEL_PARAMS,
        one_class_svm_params=ONE_CLASS_SVM_PARAMS,
    )
    X_train = np.vstack([f for _, f in train])
    model.fit(X_train, save=False)

    own_scores = np.array([model.predict(f) for _, f in test_own]).ravel()

    other_scores_list: List[np.ndarray] = []
    for other_windows in test_others.values():
        if not other_windows:
            continue
        other_scores_list.append(
            np.array([model.predict(f) for _, f in other_windows]).ravel()
        )

    other_scores = (np.concatenate(other_scores_list)
                    if other_scores_list else np.empty(0))

    if own_scores.size and other_scores.size:
        scores = np.concatenate([own_scores, other_scores])
        labels = np.concatenate([
            np.ones(own_scores.size, dtype=np.int32),
            np.zeros(other_scores.size, dtype=np.int32),
        ])
        roc_auc = float(roc_auc_score(labels, scores))
    else:
        roc_auc = float("nan")

    log.info("ROC AUC: %.4f", roc_auc)

    return {"roc_auc": roc_auc}


def _mean_median(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan"), float("nan")
    return (float(np.nanmean(arr)),
            float(np.nanmedian(arr)))


def compute_metrics(dataset_dir: str = DATASET_DIT,
                    force_recompute: bool = False,
                    window_mode: str = DEFAULT_WINDOW_MODE,
                    ) -> Dict[str, float]:
    users = get_features(dataset_dir=dataset_dir,
                         force_recompute=force_recompute,
                         window_mode=window_mode)
    if not users:
        raise RuntimeError("No users with usable data found.")

    splits: Dict[str, Tuple[UserFeatures, UserFeatures]] = {}
    for uid, windows in users.items():
        train, test = split_train_test(windows)
        if not train or not test:
            log.warning("User %s: empty train/test split, skipping", uid)
            continue
        splits[uid] = (train, test)

    if not splits:
        raise RuntimeError("No users with a valid train/test split.")

    per_user: Dict[str, Dict[str, float]] = {}
    for uid, (train, test_own) in splits.items():
        test_others = {
            other_uid: other_test
            for other_uid, (_, other_test) in splits.items()
            if other_uid != uid
        }
        try:
            per_user[uid] = evaluate_user(uid, train, test_own, test_others)
        except Exception as exc:                                # noqa: BLE001
            log.exception("Evaluation failed for user %s: %s", uid, exc)

    if not per_user:
        raise RuntimeError("No valid per-user evaluations.")

    keys = ["roc_auc"]
    agg: Dict[str, Tuple[float, float]] = {}
    for k in keys:
        agg[k] = _mean_median([m[k] for m in per_user.values()])

    summary: Dict[str, float] = {"n_users": len(per_user)}
    for k, (mean, median) in agg.items():
        summary["mean_{}".format(k)] = mean
        summary["median_{}".format(k)] = median

    def _fmt(key: str) -> str:
        mean, median = agg[key]
        return "{:10.4f}  {:10.4f}".format(mean, median)

    log.info("=" * 50)
    log.info("Evaluation summary over %d users (window_mode=%s)",
             summary["n_users"], window_mode)
    log.info("=" * 50)
    log.info("                        :       mean      median")
    log.info("-" * 50)
    log.info("ROC AUC                 : %s", _fmt("roc_auc"))
    log.info("-" * 50)

    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate mouse-dynamics features (windowed)."
    )
    p.add_argument("--dataset", default=DATASET_DIT,
                   help="dataset root directory")
    p.add_argument("--rebuild", action="store_true",
                   help="force feature re-extraction, ignoring the cache")
    p.add_argument("--window-mode", choices=sorted(WINDOW_EXTRACTORS),
                   default=DEFAULT_WINDOW_MODE,
                   help="window slicing strategy: %(choices)s "
                        "(default: %(default)s)")
    args = p.parse_args()

    compute_metrics(dataset_dir=args.dataset,
                    force_recompute=args.rebuild,
                    window_mode=args.window_mode)