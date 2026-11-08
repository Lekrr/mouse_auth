from utils.cashe import cached
import pandas as pd
import numpy as np


# ARTICLE:
#
# Mondal S., Bours P. A study on continuous authentication using a combination of
# keystroke and mouse biometrics //Neurocomputing. – 2017. – Т. 230. – С. 1-22.


def get_bin(dist: float, threshold: int = 1000) -> float:
    if dist <= threshold:
        return dist // (threshold // 5)
    elif dist <= 2 * threshold:
        return 5 + (dist - threshold) // (threshold // 10)
    elif dist <= 3 * threshold:
        return 15 + (dist - 2 * threshold) // (threshold // 4)
    else:
        return 19


def get_det(db: pd.DataFrame) -> np.ndarray:
    Pn_Po = np.array([db.x.iloc[-1] - db.x.iloc[0], db.y.iloc[-1] - db.y.iloc[0]])
    x0, y0 = db.x.iloc[0], db.y.iloc[0]
    det = np.array([np.linalg.det([Pn_Po, [x - x0, y - y0]])
                    for x, y in zip(db.x.iloc[1:].values, db.y.iloc[1:].values)])
    return det


# FEATURE EXTRACTION FUNCTION:

def direction_bin(db: pd.DataFrame, n_bin: int = 8) -> int:
    # FIX: use np.gradient (central differences, correct one-sided endpoints)
    grad_x = np.gradient(db.x.values)
    grad_y = np.gradient(db.y.values)
    direction = np.rad2deg(np.arctan2(grad_y, grad_x)) + 180
    bins = (direction / (360.0 / n_bin)).astype(int) % n_bin
    return int(np.argmax(np.bincount(bins, minlength=n_bin)))

@cached
def actual_distance(db: pd.DataFrame) -> float:
    return float(((db.x.iloc[0] - db.x.iloc[-1]) ** 2 +
                  (db.y.iloc[0] - db.y.iloc[-1]) ** 2) ** 0.5)


def actual_distance_bin(db: pd.DataFrame, threshold: int = 1000) -> float:
    # FIX: threshold=1000 (was 250) and no int() truncation
    return get_bin(actual_distance(db, cache=True), threshold=threshold)


@cached
def curve_length(db: pd.DataFrame) -> float:
    return float(np.nansum(np.hypot(db.x.diff().values, db.y.diff().values)))


def curve_length_bin(db: pd.DataFrame, threshold: int = 1000) -> float:
    return get_bin(curve_length(db, cache=True), threshold=threshold)


def length_ratio(db: pd.DataFrame) -> float:
    ad = actual_distance(db, cache=True)
    return 0.0 if ad == 0 else curve_length(db, cache=True) / ad


def actual_speed(db: pd.DataFrame) -> float:
    dt = db.time.iloc[-1] - db.time.iloc[0]
    return 0.0 if dt == 0 else actual_distance(db, cache=True) / dt


@cached
def curve_speed(db: pd.DataFrame) -> float:
    dt = db.time.diff().values[1:]
    dt[dt == 0] = 1e-6
    return float(
        (np.hypot(db.x.diff().values[1:], db.y.diff().values[1:]) / dt).mean()
    )


def curve_acceleration(db: pd.DataFrame) -> float:
    dt = db.time.iloc[-1] - db.time.iloc[0]
    return 0.0 if dt == 0 else curve_speed(db, cache=True) / dt


@cached
def mean_movement_offset(db: pd.DataFrame) -> float:
    Pn_Po = np.array([db.x.iloc[-1] - db.x.iloc[0],
                      db.y.iloc[-1] - db.y.iloc[0]])
    norm = np.linalg.norm(Pn_Po)
    return 0.0 if norm == 0 else float((get_det(db) / norm).mean())


def mean_movement_error(db: pd.DataFrame) -> float:
    Pn_Po = np.array([db.x.iloc[-1] - db.x.iloc[0],
                      db.y.iloc[-1] - db.y.iloc[0]])
    norm = np.linalg.norm(Pn_Po)
    return 0.0 if norm == 0 else float((np.abs(get_det(db) / norm)).mean())


def mean_movement_variability(db: pd.DataFrame) -> float:
    return float(
        ((db.y.iloc[1:-1].values - mean_movement_offset(db, cache=True)) ** 2)
        .mean() ** 0.5
    )


@cached
def mean_curvature(db: pd.DataFrame) -> float:
    xy = np.hypot(db.x.values, db.y.values)
    mask = (xy != 0)
    if np.all(~mask):
        return 0.0
    return float(np.mean(np.arctan2(db.y.values, db.x.values)[mask] / xy[mask]))


def mean_curvature_change_rate(db: pd.DataFrame) -> float:
    xy = ((db.x.iloc[-1] - db.x.iloc[:-1].values) ** 2 +
          (db.y.iloc[-1] - db.y.iloc[:-1].values) ** 2) ** 0.5
    mask = (xy != 0)
    if np.all(~mask):
        return 0.0
    return float(
        np.mean(np.arctan2(db.y.iloc[:-1].values,
                           db.x.iloc[:-1].values)[mask] / xy[mask])
    )


def mean_curvature_velocity(db: pd.DataFrame) -> float:
    dt = db.time.iloc[-1] - db.time.iloc[0]
    return 0.0 if dt == 0 else mean_curvature(db, cache=True) / dt


def mean_curvature_velocity_change_rate(db: pd.DataFrame) -> float:
    dt = db.time.iloc[-1] - db.time.iloc[0]
    return 0.0 if dt == 0 else mean_curvature(db, cache=True) / (dt ** 2)


def mean_angular_velocity(db: pd.DataFrame) -> float:
    a = np.array([db.x.iloc[:-2].values - db.x.iloc[1:-1].values,
                  db.y.iloc[:-2].values - db.y.iloc[1:-1].values])
    b = np.array([db.x.iloc[2:].values - db.x.iloc[1:-1].values,
                  db.y.iloc[2:].values - db.y.iloc[1:-1].values])
    norm = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    mask = (norm != 0)
    if np.all(~mask):
        return 0.0

    x = np.minimum(1, np.maximum(-1, np.sum(a * b, axis=0)[mask] / norm[mask]))
    angle = np.arccos(x)
    dt = (db.time.iloc[2:].values - db.time.iloc[:-2].values)[mask]

    valid = dt != 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(angle[valid] / dt[valid]))