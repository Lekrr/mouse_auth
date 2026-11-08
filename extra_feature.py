from utils.cashe import cached
import pandas as pd
import numpy as np

DIRECTION_BINS = 8
MOVEMENT_GAP_MS = 100
DIST_BINS = 3
SCREEN_GRID = 3

def _split_into_movements(db: pd.DataFrame,
                          max_gap: float = MOVEMENT_GAP_MS) -> list:
    if len(db) < 2:
        return []
    dt = db.time.diff().values[1:]
    breaks = np.where(dt > max_gap)[0] + 1
    boundaries = [0] + list(breaks) + [len(db)]
    segments = []
    for i in range(len(boundaries) - 1):
        seg = db.iloc[boundaries[i]:boundaries[i + 1]].reset_index(drop=True)
        if len(seg) >= 2:
            segments.append(seg)
    return segments


def _step_vectors(db: pd.DataFrame):
    dx = db.x.diff().values[1:]
    dy = db.y.diff().values[1:]
    dist = np.hypot(dx, dy)
    dt = db.time.diff().values[1:]
    return dx, dy, dist, dt


def _direction_bins(dx, dy, dist, n_bins=DIRECTION_BINS):
    mask = dist > 0
    if not np.any(mask):
        return np.array([], dtype=int), mask
    direction = (np.rad2deg(np.arctan2(dy[mask], dx[mask])) + 180.0) % 360.0
    bins = (direction / (360.0 / n_bins)).astype(int) % n_bins
    return bins, mask


def _distance_edges(dist, n_bins):
    if len(dist) == 0 or dist.sum() == 0:
        return None
    edges = np.percentile(dist, np.linspace(0, 100, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def direction_action_percent(db: pd.DataFrame,
                             n_bins: int = DIRECTION_BINS) -> np.ndarray:
    dx, dy, dist, _ = _step_vectors(db)
    bins, _ = _direction_bins(dx, dy, dist, n_bins)
    if len(bins) == 0:
        return np.zeros(n_bins)
    counts = np.bincount(bins, minlength=n_bins).astype(float)
    return counts / counts.sum()


def direction_distance_percent(db: pd.DataFrame,
                                n_bins: int = DIRECTION_BINS) -> np.ndarray:
    dx, dy, dist, _ = _step_vectors(db)
    bins, mask = _direction_bins(dx, dy, dist, n_bins)
    if len(bins) == 0:
        return np.zeros(n_bins)
    sums = np.bincount(bins, weights=dist[mask], minlength=n_bins)
    total = sums.sum()
    return sums / total if total > 0 else sums


def direction_time_percent(db: pd.DataFrame,
                            n_bins: int = DIRECTION_BINS) -> np.ndarray:
    dx, dy, dist, dt = _step_vectors(db)
    dt = np.where(dt <= 0, 1e-6, dt)
    bins, mask = _direction_bins(dx, dy, dist, n_bins)
    if len(bins) == 0:
        return np.zeros(n_bins)
    sums = np.bincount(bins, weights=dt[mask], minlength=n_bins)
    total = sums.sum()
    return sums / total if total > 0 else sums


def mean_movement_distance(db: pd.DataFrame,
                            max_gap: float = MOVEMENT_GAP_MS) -> float:
    segments = _split_into_movements(db, max_gap)
    if not segments:
        return 0.0
    lengths = [np.hypot(s.x.diff().values[1:],
                        s.y.diff().values[1:]).sum() for s in segments]
    return float(np.mean(lengths))


def speed_by_distance_distribution(db: pd.DataFrame,
                                    n_bins: int = DIST_BINS) -> np.ndarray:
    dx, dy, dist, dt = _step_vectors(db)
    dt = np.where(dt <= 0, 1e-6, dt)
    if len(dist) == 0 or dist.sum() == 0:
        return np.zeros(n_bins)
    v = dist / dt
    edges = _distance_edges(dist, n_bins)
    if edges is None:
        return np.zeros(n_bins)
    result = np.zeros(n_bins)
    for b in range(n_bins):
        sel = (dist >= edges[b]) & (dist < edges[b + 1])
        if np.any(sel):
            result[b] = v[sel].mean()
    total = result.sum()
    return result / total if total > 0 else result


def acceleration_by_distance_distribution(db: pd.DataFrame,
                                           n_bins: int = DIST_BINS) -> np.ndarray:
    dx, dy, dist, dt = _step_vectors(db)
    dt = np.where(dt <= 0, 1e-6, dt)
    if len(dist) < 2:
        return np.zeros(n_bins)
    v = dist / dt
    a = np.diff(v) / (dt[:-1] + dt[1:])
    dist_a = dist[:-1] + dist[1:]
    n = min(len(a), len(dist_a))
    a, dist_a = a[:n], dist_a[:n]
    if dist_a.sum() == 0:
        return np.zeros(n_bins)
    edges = _distance_edges(dist_a, n_bins)
    if edges is None:
        return np.zeros(n_bins)
    result = np.zeros(n_bins)
    for b in range(n_bins):
        sel = (dist_a >= edges[b]) & (dist_a < edges[b + 1])
        if np.any(sel):
            result[b] = a[sel].mean()
    total = result.sum()
    return result / total if total > 0 else result


def screen_region_distribution(db: pd.DataFrame,
                                grid: int = SCREEN_GRID) -> np.ndarray:
    n = grid * grid
    x, y = db.x.values, db.y.values
    if len(x) == 0:
        return np.zeros(n)
    xr, yr = x.max() - x.min(), y.max() - y.min()
    if xr == 0 or yr == 0:
        out = np.zeros(n); out[0] = 1.0
        return out
    xb = ((x - x.min()) / xr * grid).astype(int).clip(0, grid - 1)
    yb = ((y - y.min()) / yr * grid).astype(int).clip(0, grid - 1)
    counts = np.bincount(yb * grid + xb, minlength=n).astype(float)
    return counts / counts.sum()


def direction_max_speed(db: pd.DataFrame,
                        n_bins: int = DIRECTION_BINS) -> np.ndarray:
    dx, dy, dist, dt = _step_vectors(db)
    dt = np.where(dt <= 0, 1e-6, dt)
    bins, mask = _direction_bins(dx, dy, dist, n_bins)
    if len(bins) == 0:
        return np.zeros(n_bins)
    v = (dist / dt)[mask]
    result = np.zeros(n_bins)
    for b in range(n_bins):
        sel = bins == b
        if np.any(sel):
            result[b] = v[sel].max()
    return result


def direction_min_speed(db: pd.DataFrame,
                        n_bins: int = DIRECTION_BINS) -> np.ndarray:
    dx, dy, dist, dt = _step_vectors(db)
    dt = np.where(dt <= 0, 1e-6, dt)
    bins, mask = _direction_bins(dx, dy, dist, n_bins)
    if len(bins) == 0:
        return np.zeros(n_bins)
    v = (dist / dt)[mask]
    result = np.zeros(n_bins)
    for b in range(n_bins):
        sel = bins == b
        if np.any(sel):
            result[b] = v[sel].min()
    return result



def time_by_distance_distribution(db: pd.DataFrame,
                                   n_bins: int = DIST_BINS) -> np.ndarray:
    dx, dy, dist, dt = _step_vectors(db)
    dt = np.where(dt <= 0, 1e-6, dt)
    if len(dist) == 0 or dist.sum() == 0:
        return np.zeros(n_bins)
    edges = _distance_edges(dist, n_bins)
    if edges is None:
        return np.zeros(n_bins)
    result = np.zeros(n_bins)
    for b in range(n_bins):
        sel = (dist >= edges[b]) & (dist < edges[b + 1])
        if np.any(sel):
            result[b] = dt[sel].sum()
    total = result.sum()
    return result / total if total > 0 else result



def movement_direction(db: pd.DataFrame) -> float:
    dx, dy, dist, _ = _step_vectors(db)
    mask = dist > 0
    if not np.any(mask):
        return 0.0
    direction = np.rad2deg(np.arctan2(dy[mask], dx[mask])) + 180.0
    return float(direction.mean())



def bend_angle(db: pd.DataFrame) -> float:
    if len(db) < 3:
        return 0.0
    x, y = db.x.values, db.y.values
    dx1, dy1 = np.diff(x)[:-1], np.diff(y)[:-1]
    dx2, dy2 = np.diff(x)[1:], np.diff(y)[1:]
    mask = ((dx1 ** 2 + dy1 ** 2) > 0) & ((dx2 ** 2 + dy2 ** 2) > 0)
    if not np.any(mask):
        return 0.0
    a1 = np.arctan2(dy1[mask], dx1[mask])
    a2 = np.arctan2(dy2[mask], dx2[mask])
    return float(np.mod(np.rad2deg(a2 - a1), 360.0).mean())



def bend_depth(db: pd.DataFrame) -> float:
    if len(db) < 3:
        return 0.0
    x, y = db.x.values, db.y.values
    p1 = np.column_stack([x[:-2], y[:-2]])
    p2 = np.column_stack([x[1:-1], y[1:-1]])
    p3 = np.column_stack([x[2:], y[2:]])
    seg = p3 - p1
    seg_len = np.linalg.norm(seg, axis=1)
    mask = seg_len > 0
    if not np.any(mask):
        return 0.0
    v = p2[mask] - p1[mask]
    cross = (np.abs(seg[mask, 0] * v[:, 1] - seg[mask, 1] * v[:, 0])
             / seg_len[mask])
    valid = cross > 0
    if not np.any(valid):
        return 0.0
    return float((seg_len[mask][valid] / cross[valid]).mean())



def trajectory_points_count(db: pd.DataFrame,
                            max_gap: float = MOVEMENT_GAP_MS) -> float:
    segments = _split_into_movements(db, max_gap)
    if not segments:
        return 0.0
    return float(np.mean([len(s) for s in segments]))



def trajectory_time(db: pd.DataFrame,
                    max_gap: float = MOVEMENT_GAP_MS) -> float:
    segments = _split_into_movements(db, max_gap)
    if not segments:
        return 0.0
    return float(np.mean([s.time.iloc[-1] - s.time.iloc[0]
                          for s in segments]))



def trajectory_asymmetry(db: pd.DataFrame) -> float:
    if len(db) < 2:
        return 0.5
    x, y = db.x.values, db.y.values
    sx, sy, ex, ey = x[0], y[0], x[-1], y[-1]
    line = np.array([ex - sx, ey - sy])
    if np.linalg.norm(line) == 0:
        return 0.5
    left_len = total_len = 0.0
    for i in range(1, len(x)):
        seg = np.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
        total_len += seg
        mx, my = 0.5 * (x[i] + x[i - 1]), 0.5 * (y[i] + y[i - 1])
        if line[0] * (my - sy) - line[1] * (mx - sx) > 0:
            left_len += seg
    return left_len / total_len if total_len > 0 else 0.5