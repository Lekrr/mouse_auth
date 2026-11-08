import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from tester import (
    get_features, split_train_test, evaluate_user, DATASET_DIT,
    UserFeatures, DEFAULT_WINDOW_MODE,
)
from window_extraction import WINDOW_EXTRACTORS


from feature_extraction import (
    direction_bin, actual_distance, actual_distance_bin,
    curve_length, curve_length_bin, length_ratio, actual_speed, curve_speed,
    curve_acceleration, mean_movement_offset, mean_movement_error,
    mean_movement_variability, mean_curvature, mean_curvature_change_rate,
    mean_curvature_velocity, mean_curvature_velocity_change_rate, mean_angular_velocity,

    min_x, max_x, mean_x, std_x, max_min_x,
    min_y, max_y, mean_y, std_y, max_min_y,
    min_vx, max_vx, mean_vx, std_vx, max_min_vx,
    min_vy, max_vy, mean_vy, std_vy, max_min_vy,
    min_v, max_v, mean_v, std_v, max_min_v,
    min_a, max_a, mean_a, std_a, max_min_a,
    min_j, max_j, mean_j, std_j, max_min_j,
    min_am, max_am, mean_am, std_am, max_min_am,
    min_av, max_av, mean_av, std_av, max_min_av,
    min_c, max_c, mean_c, std_c, max_min_c,
    min_ccr, max_ccr, mean_ccr, std_ccr, max_min_ccr,
    duration_of_movement, straightness,
    TCM, SC, M3, M4, TCrv, VCrv,

    direction_action_percent, direction_distance_percent, direction_time_percent,
    mean_movement_distance,
    speed_by_distance_distribution, acceleration_by_distance_distribution,
    screen_region_distribution,
    direction_max_speed, direction_min_speed,
    time_by_distance_distribution,
    movement_direction, bend_angle, bend_depth,
    trajectory_points_count, trajectory_time, trajectory_asymmetry,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("feature_select")

FEATURE_FUNCTIONS = [
    direction_bin, actual_distance, actual_distance_bin,
    curve_length, curve_length_bin, length_ratio, actual_speed, curve_speed,
    curve_acceleration, mean_movement_offset, mean_movement_error,
    mean_movement_variability, mean_curvature, mean_curvature_change_rate,
    mean_curvature_velocity, mean_curvature_velocity_change_rate, mean_angular_velocity,

    # min_x, max_x, mean_x, std_x, max_min_x,
    # min_y, max_y, mean_y, std_y, max_min_y,
    min_vx, max_vx, mean_vx, std_vx, max_min_vx,
    min_vy, max_vy, mean_vy, std_vy, max_min_vy,
    min_v, max_v, mean_v, std_v, max_min_v,
    min_a, max_a, mean_a, std_a, max_min_a,
    min_j, max_j, mean_j, std_j, max_min_j,
    min_am, max_am, mean_am, std_am, max_min_am,
    min_av, max_av, mean_av, std_av, max_min_av,
    min_c, max_c, mean_c, std_c, max_min_c,
    min_ccr, max_ccr, mean_ccr, std_ccr, max_min_ccr,
    # duration_of_movement,
    straightness,
    TCM, SC, M3, M4, TCrv, VCrv,

    direction_action_percent, direction_distance_percent, direction_time_percent,
    mean_movement_distance,
    speed_by_distance_distribution, acceleration_by_distance_distribution,
    screen_region_distribution,
    direction_max_speed, direction_min_speed,
    time_by_distance_distribution,
    movement_direction, bend_angle, bend_depth,
    trajectory_points_count, trajectory_time, trajectory_asymmetry,
]

GBM_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
)


def resolve_feature_names(total_dim: int) -> List[str]:
    """Best-effort: probe extractor functions on a synthetic df."""
    rng = np.random.RandomState(0)
    n = 100
    df = pd.DataFrame({
        "time": np.arange(n) * 0.01,
        "x": np.cumsum(rng.randn(n) * 5),
        "y": np.cumsum(rng.randn(n) * 5),
    })
    try:
        from utils.cashe import clear_cached
    except Exception:
        clear_cached = lambda: None

    names: List[str] = []
    for f in FEATURE_FUNCTIONS:
        try:
            clear_cached()
            r = np.asarray(f(df)).ravel()
            base = f.__name__
            if r.size == 1:
                names.append(base)
            else:
                names.extend(f"{base}[{i}]" for i in range(r.size))
        except Exception as exc:
            log.debug("Probe failed for %s: %s", f.__name__, exc)
            names.append(f.__name__)

    if len(names) != total_dim:
        log.warning(
            "Probed %d names but feature vectors have %d dims — "
            "using numbered names. Add a `feature_names()` helper to "
            "feature_extraction.py if you want readable labels.",
            len(names), total_dim,
        )
        return [f"f{i}" for i in range(total_dim)]
    return names


def _sample_negatives(splits: Dict[str, Tuple[UserFeatures, UserFeatures]],
                      uid: str, which: str, cap: int,
                      rng: np.random.RandomState) -> np.ndarray:
    chunks = []
    for other_uid, (train_w, test_w) in splits.items():
        if other_uid == uid:
            continue
        flat = train_w if which == "train" else test_w
        if not flat:
            continue
        arr = np.vstack([f for _, f in flat])
        if cap and len(arr) > cap:
            arr = arr[rng.choice(len(arr), cap, replace=False)]
        chunks.append(arr)
    return np.vstack(chunks) if chunks else np.empty((0, 0))


def gbm_for_user(uid: str,
                 splits: Dict[str, Tuple[UserFeatures, UserFeatures]],
                 neg_cap: int,
                 seed: int) -> Optional[Dict]:
    train_w, test_w = splits[uid]
    if len(train_w) < 10 or len(test_w) < 10:
        return None

    rng = np.random.RandomState(seed)
    Xtp = np.vstack([f for _, f in train_w])
    Xep = np.vstack([f for _, f in test_w])
    Xtn = _sample_negatives(splits, uid, "train", neg_cap, rng)
    Xen = _sample_negatives(splits, uid, "test", neg_cap, rng)
    if not len(Xtn) or not len(Xen):
        return None

    X_train = np.vstack([Xtp, Xtn])
    y_train = np.concatenate([np.ones(len(Xtp)), np.zeros(len(Xtn))])
    X_test = np.vstack([Xep, Xen])
    y_test = np.concatenate([np.ones(len(Xep)), np.zeros(len(Xen))])

    n_pos, n_neg = int((y_train == 1).sum()), int((y_train == 0).sum())
    w = np.ones(len(y_train))
    if n_pos and n_neg:
        w[y_train == 0] = n_pos / n_neg

    clf = GradientBoostingClassifier(random_state=seed, **GBM_PARAMS)
    clf.fit(X_train, y_train, sample_weight=w)
    auc = float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

    return {
        "importances": clf.feature_importances_.copy(),
        "auc": auc,
        "n_train_pos": n_pos,
        "n_train_neg": n_neg,
    }


def aggregate(per_user: Dict[str, Dict], feature_names: List[str]) -> pd.DataFrame:
    uids = list(per_user.keys())
    imps = np.vstack([per_user[u]["importances"] for u in uids])
    aucs = np.array([per_user[u]["auc"] for u in uids])

    ranks = np.argsort(np.argsort(-imps, axis=1), axis=1) + 1

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_importance": imps.mean(axis=0),
        "std_importance": imps.std(axis=0),
        "mean_rank": ranks.mean(axis=0),
        "median_rank": np.median(ranks, axis=0),
    }).sort_values("mean_rank").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["cumulative_importance"] = df["mean_importance"].cumsum()

    log.info("Per-user GBM ROC AUC: mean=%.4f median=%.4f (n=%d)",
             aucs.mean(), np.median(aucs), len(aucs))
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=DATASET_DIT)
    p.add_argument("--rebuild", action="store_true",
                   help="force feature re-extraction (ignore cache)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=30,
                   help="how many top features to print")
    p.add_argument("--neg-cap", type=int, default=100,
                   help="max negatives sampled per other user (train/test each)")
    p.add_argument("--out", default="feature_ranking.csv")
    p.add_argument("--window-mode", choices=sorted(WINDOW_EXTRACTORS),
                   default=DEFAULT_WINDOW_MODE,
                   help="window slicing strategy: %(choices)s "
                        "(default: %(default)s)")
    args = p.parse_args()

    users = get_features(dataset_dir=args.dataset,
                         force_recompute=args.rebuild,
                         window_mode=args.window_mode,
                         extraction_function=FEATURE_FUNCTIONS)
    if len(users) < 2:
        raise RuntimeError("Need at least 2 users with usable data.")

    any_uid = next(iter(users))
    total_dim = users[any_uid][0][1].size
    feature_names = resolve_feature_names(total_dim)
    log.info("Feature vector length: %d (names resolved: %s)",
             total_dim, feature_names[0] if feature_names else "?")

    splits: Dict[str, Tuple[UserFeatures, UserFeatures]] = {}
    for uid, windows in users.items():
        tr, te = split_train_test(windows, seed=args.seed)
        if not tr or not te:
            log.warning("Skipping user %s (no valid split)", uid)
            continue
        splits[uid] = (tr, te)
    if len(splits) < 2:
        raise RuntimeError("Need at least 2 users with valid splits.")

    per_user: Dict[str, Dict] = {}
    for uid in splits:
        r = gbm_for_user(uid, splits, args.neg_cap, args.seed)
        if r is None:
            log.warning("Skipping user %s (insufficient data)", uid)
            continue
        per_user[uid] = r
        log.info("User %-12s GBM AUC=%.4f  pos=%d neg=%d",
                 uid, r["auc"], r["n_train_pos"], r["n_train_neg"])
    if not per_user:
        raise RuntimeError("No user had enough data for GBM.")

    ranking = aggregate(per_user, feature_names)
    ranking.to_csv(args.out, index=False)
    log.info("Wrote full ranking to %s", args.out)

    k = min(args.top_k, len(ranking))
    log.info("=" * 66)
    log.info("Top %d features by mean per-user rank", k)
    log.info("=" * 66)
    log.info("%4s  %-32s %10s %10s %10s",
             "Rank", "Feature", "MeanImp", "MeanRank", "CumImp")
    log.info("-" * 66)
    for _, r in ranking.head(k).iterrows():
        log.info("%4d  %-32s %10.4f %10.2f %10.4f",
                 int(r["rank"]), r["feature"],
                 r["mean_importance"], r["mean_rank"],
                 r["cumulative_importance"])
