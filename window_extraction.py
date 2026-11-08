import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config import RawData, MIN_N_ACTION
from src.pipeline.preprocessing import preprocessing
from feature_extraction import extractor

log = logging.getLogger(__name__)

UserFeatures = List[Tuple[float, np.ndarray]]

MIN_WINDOW_SEC: float = 60.0
MAX_WINDOW_SEC: float = 540.0

MIN_EVENTS: int = 3000
MAX_EVENTS: int = 5000

PAUSE_SPLIT_SEC: float = 3.0
WINDOW_OVERLAP: float = 0.25


def _extract_window(window_data: np.ndarray,
                    min_n_action: int,
                    extraction_function=None
                    ) -> Optional[Tuple[float, np.ndarray]]:
    if window_data.ndim != 2 or window_data.shape[0] < max(min_n_action, 3):
        return None

    try:
        feature = extractor(window_data, extraction_function=extraction_function)
    except Exception as exc:                                    # noqa: BLE001
        log.warning("Extractor failed on a window: %s", exc)
        return None

    feature = np.asarray(feature, dtype=np.float64).ravel()
    if not np.all(np.isfinite(feature)):
        log.warning("Skipping window with non-finite features")
        return None

    return float(window_data[0, 0]), feature


def _load_preprocessed(mouse_csv: str, min_n_action: int) -> Optional[np.ndarray]:
    try:
        df = pd.read_csv(mouse_csv, usecols=["time", "x_pos", "y_pos"])
    except Exception as exc:                                    # noqa: BLE001
        log.warning("Cannot read %s: %s", mouse_csv, exc)
        return None
    if df.empty:
        return None

    parsed = pd.to_datetime(
        df["time"], format="%Y-%m-%dT%H:%M:%S.%f", errors="coerce",
    )

    bad_mask = parsed.isna()
    if bad_mask.any():
        bad_idx = np.flatnonzero(bad_mask.to_numpy())
        bad_lines = (bad_idx + 2).tolist()
        bad_values = df.loc[bad_mask, "time"].tolist()

        log.warning(
            "Skipping %d bad row(s) in %s at CSV line(s) %s",
            len(bad_lines), mouse_csv, bad_lines,
        )
        for line_no, raw_val in zip(bad_lines, bad_values):
            log.warning("    line %d: time=%r", line_no, raw_val)

        df = df.loc[~bad_mask].reset_index(drop=True)
        parsed = parsed.loc[~bad_mask].reset_index(drop=True)

    if df.empty:
        log.warning("No valid rows left in %s", mouse_csv)
        return None

    df["time"] = parsed.astype("int64") / 1e9

    raw_data: List[RawData] = [
        RawData(timestamp=float(t), button=0, state=0, x=float(x), y=float(y))
        for t, x, y in zip(df["time"], df["x_pos"], df["y_pos"])
    ]
    if not raw_data:
        return None

    try:
        data = preprocessing(raw_data, min_n_action=min_n_action)
    except Exception as exc:                                    # noqa: BLE001
        log.warning("Preprocessing failed on %s: %s", mouse_csv, exc)
        return None

    if data.ndim != 2 or data.shape[0] < max(min_n_action, 3):
        log.warning(
            "%s: preprocessing produced too few events (%d) from %d raw events",
            mouse_csv,
            data.shape[0] if data.ndim == 2 else 0,
            len(raw_data),
        )
        return None

    return data


def _split_segments(data: np.ndarray, pause_split_sec: float) -> List[np.ndarray]:
    segments: List[np.ndarray] = []
    seg_start_idx = 0
    for i in range(1, data.shape[0]):
        if data[i, 0] - data[i - 1, 0] > pause_split_sec:
            segments.append(data[seg_start_idx:i])
            seg_start_idx = i
    segments.append(data[seg_start_idx:])
    return segments


def windows_by_time(mouse_csv: str,
                    min_n_action: int,
                    min_window_sec: float = MIN_WINDOW_SEC,
                    max_window_sec: float = MAX_WINDOW_SEC,
                    pause_split_sec: float = PAUSE_SPLIT_SEC,
                    overlap: float = WINDOW_OVERLAP,
                    extraction_function=None,
                    ) -> UserFeatures:
    data = _load_preprocessed(mouse_csv, min_n_action)
    if data is None:
        return []

    segments = _split_segments(data, pause_split_sec)

    step = max_window_sec * (1.0 - overlap)
    if step <= 0.0:
        step = max_window_sec

    windows: UserFeatures = []
    seg_durations: List[float] = []

    for seg in segments:
        seg_start = float(seg[0, 0])
        seg_end = float(seg[-1, 0])
        seg_duration = seg_end - seg_start
        seg_durations.append(seg_duration)

        if seg_duration >= min_window_sec:
            t_start = seg_start
            while True:
                t_end = min(t_start + max_window_sec, seg_end)
                if t_end - t_start < min_window_sec:
                    break

                mask = (seg[:, 0] >= t_start) & (seg[:, 0] <= t_end)
                win = _extract_window(
                    seg[mask], min_n_action,
                    extraction_function=extraction_function,
                )
                if win is not None:
                    windows.append(win)

                if t_end >= seg_end:
                    break
                t_start += step

    if not windows:
        log.warning(
            "%s: 0 windows from %d preprocessed events | %d segments "
            "| segment dur min=%.1fs median=%.1fs max=%.1fs",
            mouse_csv, data.shape[0], len(segments),
            min(seg_durations) if seg_durations else 0.0,
            float(np.median(seg_durations)) if seg_durations else 0.0,
            max(seg_durations) if seg_durations else 0.0,
        )

    return windows


def windows_by_events(mouse_csv: str,
                      min_n_action: int,
                      min_events: int = MIN_EVENTS,
                      max_events: int = MAX_EVENTS,
                      pause_split_sec: float = PAUSE_SPLIT_SEC,
                      overlap: float = WINDOW_OVERLAP,
                      extraction_function=None,
                      ) -> UserFeatures:
    if max_events <= 0 or min_events <= 0:
        raise ValueError("min_events and max_events must be positive")
    if min_events > max_events:
        raise ValueError("min_events must be <= max_events")

    data = _load_preprocessed(mouse_csv, min_n_action)
    if data is None:
        return []

    segments = _split_segments(data, pause_split_sec)

    step = max(1, int(round(max_events * (1.0 - overlap))))

    windows: UserFeatures = []
    seg_lengths: List[int] = []

    for seg in segments:
        n = seg.shape[0]
        seg_lengths.append(n)

        if n >= min_events:
            start = 0
            while True:
                end = min(start + max_events, n)
                if end - start < min_events:
                    break

                win = _extract_window(
                    seg[start:end], min_n_action,
                    extraction_function=extraction_function,
                )
                if win is not None:
                    windows.append(win)

                if end >= n:
                    break
                start += step

    if not windows:
        log.warning(
            "%s: 0 windows from %d preprocessed events | %d segments "
            "| segment events min=%d median=%d max=%d "
            "| min_events=%d max_events=%d",
            mouse_csv, data.shape[0], len(segments),
            min(seg_lengths) if seg_lengths else 0,
            int(np.median(seg_lengths)) if seg_lengths else 0,
            max(seg_lengths) if seg_lengths else 0,
            min_events, max_events,
        )

    return windows


WINDOW_EXTRACTORS = {
    "time": windows_by_time,
    "events": windows_by_events,
}
