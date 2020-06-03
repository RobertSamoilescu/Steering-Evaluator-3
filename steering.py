import numpy as np

CAR_L = 2.634   # Wheel base
CAR_T = 1.497  # Tread

MIN_TURNING_RADIUS = 5.
MAX_STEER = 500
MAX_WHEEL_ANGLE = np.rad2deg(np.arctan(CAR_L / MIN_TURNING_RADIUS))
STEERING_RATIO = MAX_STEER / MAX_WHEEL_ANGLE
eps = 1e-12


def get_delta_from_steer(steer, steering_ratio=STEERING_RATIO):
    """
    :param steer:  car steer in degrees
    :param steering_ratio: ratio of maximum steer and maximum wheel angle (is constant)
    :return: wheel angle
    """
    sgn = np.sign(steer)
    delta = sgn * min(MAX_WHEEL_ANGLE, abs(steer)/steering_ratio)
    return delta


def get_steer_from_delta(delta, steering_ratio=STEERING_RATIO):
    """
    :param delta: wheel angle
    :param steering_ratio: ratio of maximum steer and maximum wheel angle (is constant)
    :return:
    """
    sgn = np.sign(delta)
    steer = sgn * min(MAX_STEER, abs(delta) * steering_ratio)
    return steer


def get_radius_from_delta(delta, car_l=CAR_L):
    """
    :param delta: wheel angle
    :param car_l: wheel base
    :return: radius of the circle that the car makes
    """
    sgn = np.sign(delta)
    r = car_l/np.tan(np.deg2rad(abs(delta), dtype=np.float32) + eps)
    r = sgn * max(r, MIN_TURNING_RADIUS)
    return r


def get_delta_from_radius(r, car_l=CAR_L, car_t=CAR_T):
    """
    :param r: Turn radius ( calculated against back center)
    :param car_l: Wheel base
    :param car_t: Tread
    :return: Angles of front center, inner wheel, outer wheel
    """
    sgn = np.sign(r)
    r = max(abs(r), MIN_TURNING_RADIUS)
    delta_i = sgn * np.rad2deg(np.arctan(car_l / (r - car_t / 2.)))
    delta = sgn * np.rad2deg(np.arctan(car_l / r))
    delta_o = sgn * np.rad2deg(np.arctan(car_l / (r + car_t / 2.)))
    return delta, delta_i, delta_o
