import numpy as np
import pandas as pd

from my_feature import *
from ars_feature import *
from extra_feature import (
    direction_action_percent, direction_distance_percent, direction_time_percent,
    mean_movement_distance,
    speed_by_distance_distribution, acceleration_by_distance_distribution,
    screen_region_distribution,
    direction_max_speed, direction_min_speed,
    time_by_distance_distribution,
    movement_direction, bend_angle, bend_depth,
    trajectory_points_count, trajectory_time, trajectory_asymmetry,
)
from utils.cashe import clear_cached


all_features = list([
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
])

def take(fn, idx):
    idx = np.asarray(idx, dtype=int)
    def wrapper(df):
        return np.asarray(fn(df)).ravel()[idx]
    wrapper.__name__ = f"{fn.__name__}_{list(idx)}"
    return wrapper

def extractor(data: np.ndarray, extraction_function = None) -> np.ndarray:
    if extraction_function is None:
        extraction_function = list([
            # mean_angular_velocity, M4, M3, TCM, min_v, min_am, VCrv, mean_av, speed_by_distance_distribution, bend_angle,
            # direction_action_percent, min_c, trajectory_points_count, mean_j, curve_acceleration, mean_movement_variability,
            # mean_am, direction_time_percent, mean_a, time_by_distance_distribution, max_av, bend_depth, std_av, TCrv,
            # direction_min_speed, max_am, max_min_av,

            mean_angular_velocity, screen_region_distribution, direction_time_percent, time_by_distance_distribution, direction_action_percent, bend_angle,
            min_am, acceleration_by_distance_distribution, mean_am, speed_by_distance_distribution, mean_curvature, direction_distance_percent,
            mean_c, VCrv, movement_direction, std_am, mean_curvature_change_rate,
        ])

    df = pd.DataFrame(data, columns=["time", "x", "y"])
    res = []

    clear_cached()

    for f in extraction_function:
        res.append(f(df))

    return np.hstack(res)