from utils.cashe import cached
import pandas as pd
import numpy as np

MS = 1e-6

# ARTICLE:
#
# Feher C. et al. User identity verification via mouse dynamics
# //Information Sciences. – 2012. – Ò. 201. – Ñ. 19-36.


# feature extraction function template:
#
# def feature_name(db: pd.DataFrame) -> float:
#     """
#     extract features <features_name> from the database <db>
#     :param db: database segment
#     :return: feature value
#     """

def min_x(db: pd.DataFrame) -> float:
    return db.x.min()


def max_x(db: pd.DataFrame) -> float:
    return db.x.max()


def mean_x(db: pd.DataFrame) -> float:
    return db.x.mean()


def std_x(db: pd.DataFrame) -> float:
    return db.x.std()


def max_min_x(db: pd.DataFrame) -> float:
    return max_x(db) - min_x(db)


def min_y(db: pd.DataFrame) -> float:
    return db.y.min()


def max_y(db: pd.DataFrame) -> float:
    return db.y.max()


def mean_y(db: pd.DataFrame) -> float:
    return db.y.mean()


def std_y(db: pd.DataFrame) -> float:
    return db.y.std()


def max_min_y(db: pd.DataFrame) -> float:
    return max_y(db) - min_y(db)


@cached
def velocity_x(db: pd.DataFrame) -> np.ndarray:
    dt = db.time.diff().values[1:]
    dt[dt == 0] = MS
    return db.x.diff().values[1:] / dt


def min_vx(db: pd.DataFrame) -> float:
    return velocity_x(db).min()


def max_vx(db: pd.DataFrame) -> float:
    return velocity_x(db, cache=True).max()


def mean_vx(db: pd.DataFrame) -> float:
    return velocity_x(db, cache=True).mean()


def std_vx(db: pd.DataFrame) -> float:
    return velocity_x(db, cache=True).std()


def max_min_vx(db: pd.DataFrame) -> float:
    return max_vx(db) - min_vx(db)


@cached
def velocity_y(db: pd.DataFrame) -> np.ndarray:
    dt = db.time.diff().values[1:]
    dt[dt == 0] = MS
    return db.y.diff().values[1:] / dt


def min_vy(db: pd.DataFrame) -> float:
    return velocity_y(db).min()


def max_vy(db: pd.DataFrame) -> float:
    return velocity_y(db, cache=True).max()


def mean_vy(db: pd.DataFrame) -> float:
    return velocity_y(db, cache=True).mean()


def std_vy(db: pd.DataFrame) -> float:
    return velocity_y(db, cache=True).std()


def max_min_vy(db: pd.DataFrame) -> float:
    return max_vy(db) - min_vy(db)


@cached
def velocity(db: pd.DataFrame) -> np.ndarray:
    return np.hypot(velocity_x(db, cache=True), velocity_y(db, cache=True))


def min_v(db: pd.DataFrame) -> float:
    return velocity(db).min()


def max_v(db: pd.DataFrame) -> float:
    return velocity(db, cache=True).max()


def mean_v(db: pd.DataFrame) -> float:
    return velocity(db, cache=True).mean()


def std_v(db: pd.DataFrame) -> float:
    return velocity(db, cache=True).std()


def max_min_v(db: pd.DataFrame) -> float:
    return max_v(db) - min_v(db)


@cached
def acceleration(db: pd.DataFrame) -> np.ndarray:
    dt = db.time.diff().values[2:]
    dt[dt == 0] = MS
    v = velocity(db, cached=True)
    dv = np.diff(v)
    return dv / dt


def acceleration_x(db: pd.DataFrame) -> np.ndarray:
    dt = db.time.diff().values[2:]
    dt[dt == 0] = MS
    v = velocity_x(db, cached=True)
    dv = np.diff(v)
    return dv / dt


def acceleration_y(db: pd.DataFrame) -> np.ndarray:
    dt = db.time.diff().values[2:]
    dt[dt == 0] = MS
    v = velocity_y(db, cached=True)
    dv = np.diff(v)
    return dv / dt


def min_a(db: pd.DataFrame) -> float:
    return acceleration(db).min()


def max_a(db: pd.DataFrame) -> float:
    return acceleration(db, cache=True).max()


def mean_a(db: pd.DataFrame) -> float:
    return acceleration(db, cache=True).mean()


def std_a(db: pd.DataFrame) -> float:
    return acceleration(db, cache=True).std()


def max_min_a(db: pd.DataFrame) -> float:
    return max_a(db) - min_a(db)


@cached
def jerk(db: pd.DataFrame) -> np.ndarray:
    dt = db.time.diff().values[3:]
    dt[dt == 0] = MS
    a = acceleration(db, cached=True)
    da = np.diff(a)
    return da / dt


def min_j(db: pd.DataFrame) -> float:
    return jerk(db).min()


def max_j(db: pd.DataFrame) -> float:
    return jerk(db, cache=True).max()


def mean_j(db: pd.DataFrame) -> float:
    return jerk(db, cache=True).mean()


def std_j(db: pd.DataFrame) -> float:
    return jerk(db, cache=True).std()


def max_min_j(db: pd.DataFrame) -> float:
    return max_j(db) - min_j(db)


@cached
def angle_of_movement(db: pd.DataFrame) -> np.ndarray:
    return np.arctan2(db.y, db.x)


def min_am(db: pd.DataFrame) -> float:
    return angle_of_movement(db).min()


def max_am(db: pd.DataFrame) -> float:
    return angle_of_movement(db, cache=True).max()


def mean_am(db: pd.DataFrame) -> float:
    return angle_of_movement(db, cache=True).mean()


def std_am(db: pd.DataFrame) -> float:
    return angle_of_movement(db, cache=True).std()


def max_min_am(db: pd.DataFrame) -> float:
    return max_am(db) - min_am(db)


@cached
def curvature(db: pd.DataFrame) -> np.ndarray:
    return angle_of_movement(db, cache=True) / np.hypot(db.x, db.y)


def min_c(db: pd.DataFrame) -> float:
    return curvature(db).min()


def max_c(db: pd.DataFrame) -> float:
    return curvature(db, cache=True).max()


def mean_c(db: pd.DataFrame) -> float:
    return curvature(db, cache=True).mean()


def std_c(db: pd.DataFrame) -> float:
    return curvature(db, cache=True).std()


def max_min_c(db: pd.DataFrame) -> float:
    return max_c(db) - min_c(db)


@cached
def curvature_change_rate(db: pd.DataFrame) -> np.ndarray:
    return curvature(db, cache=True) / np.hypot(db.x, db.y)


def min_ccr(db: pd.DataFrame) -> float:
    return curvature_change_rate(db).min()


def max_ccr(db: pd.DataFrame) -> float:
    return curvature_change_rate(db, cache=True).max()


def mean_ccr(db: pd.DataFrame) -> float:
    return curvature_change_rate(db, cache=True).mean()


def std_ccr(db: pd.DataFrame) -> float:
    return curvature_change_rate(db, cache=True).std()


def max_min_ccr(db: pd.DataFrame) -> float:
    return max_ccr(db) - min_ccr(db)


@cached
def angular_velocity(db: pd.DataFrame) -> np.ndarray:
    dt = db.time.diff().values[1:]
    dt[dt == 0] = MS
    return np.arctan2(db.y.iloc[1:], db.x.iloc[1:]) / dt


def min_av(db: pd.DataFrame) -> float:
    return angular_velocity(db).min()


def max_av(db: pd.DataFrame) -> float:
    return angular_velocity(db, cache=True).max()


def mean_av(db: pd.DataFrame) -> float:
    return angular_velocity(db, cache=True).mean()


def std_av(db: pd.DataFrame) -> float:
    return angular_velocity(db, cache=True).std()


def max_min_av(db: pd.DataFrame) -> float:
    return max_av(db) - min_av(db)


def duration_of_movement(db: pd.DataFrame) -> float:
    return db.time.iloc[-1] - db.time.iloc[0]


@cached
def traveled_distance(db: pd.DataFrame) -> float:
    return np.hypot(db.x.diff().values, db.y.diff().values)[1:].sum()


def straightness(db: pd.DataFrame) -> float:
    return 0.0 if (traveled_distance(db) == 0.0) else \
        ((db.x.iloc[0] - db.x.iloc[-1]) ** 2 +
         (db.y.iloc[0] - db.y.iloc[-1]) ** 2) ** 0.5 / traveled_distance(db, cache=True)


@cached
def TCM(db: pd.DataFrame) -> float:  # Trajectory Center of Mass
    return 0.0 if (traveled_distance(db, cache=True) == 0.0) else \
        (np.hypot(db.x.diff().values, db.y.diff().values)[1:] *
         db.time.values[1:]).sum() / traveled_distance(db, cache=True)


def SC(db: pd.DataFrame) -> float:  # Scattering Coefficient
    return 0.0 if (traveled_distance(db, cache=True) == 0.0) else \
        (np.hypot(db.x.diff().values, db.y.diff().values)[1:] *
         db.time.values[1:] ** 2).sum() / traveled_distance(db, cache=True)\
        - TCM(db, cache=True) ** 2


def M3(db: pd.DataFrame) -> float:  # Third Moment
    return 0.0 if (traveled_distance(db, cache=True) == 0.0) else \
        (np.hypot(db.x.diff().values, db.y.diff().values)[1:] *
         db.time.values[1:] ** 3).sum() / traveled_distance(db, cache=True)


def M4(db: pd.DataFrame) -> float:  # Fourth Moment
    return 0.0 if (traveled_distance(db, cache=True) == 0.0) else \
        (np.hypot(db.x.diff().values, db.y.diff().values)[1:] *
         db.time.values[1:] ** 4).sum() / traveled_distance(db, cache=True)


# def TCrv(db: pd.DataFrame) -> float:  # Trajectory Curvature
#     mask = (velocity(db, cache=True) != 0)
#     return 0.0 if np.all(~mask) else \
#         ((velocity_x(db, cache=True) * acceleration_y(db) -
#           velocity_y(db, cache=True) * acceleration_x(db))[mask] /
#          (velocity(db, cache=True) ** 3)[mask]).mean()

def TCrv(db: pd.DataFrame) -> float:  # Trajectory Curvature
    vx = velocity_x(db, cache=True)[1:]
    vy = velocity_y(db, cache=True)[1:]
    v  = velocity(db,   cache=True)[1:]
    ax = acceleration_x(db)
    ay = acceleration_y(db)

    mask = v != 0
    if np.all(~mask):
        return 0.0
    return ((vx * ay - vy * ax)[mask] / (v ** 3)[mask]).mean()


# def VCrv(db: pd.DataFrame) -> float:  # Velocity Curvature
#     return (jerk(db, cache=True) / (1 + acceleration(db, cache=True) ** 2) ** (3 / 2)).mean()

def VCrv(db: pd.DataFrame) -> float:  # Velocity Curvature
    a = acceleration(db, cache=True)[1:]
    j = jerk(db, cache=True)
    return (j / (1 + a ** 2) ** (3 / 2)).mean()
