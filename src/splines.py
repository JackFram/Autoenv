import numpy as np
import math
from scipy.sparse import linalg, lil_matrix
import tqdm

'''
used to interpolate and smooth the roadway
'''


def _fit_open(pts):
    d, n = pts.shape
    n -= 1
    Y = np.array([None for _ in range(n+1)], dtype='float')
    M = np.zeros((n+1, n+1))
    for i in range(n):
        M[i, i] = 4.0
        M[i, i + 1] = 1.0
        M[i + 1, i] = 1.0
    M[n, n] = 2.0
    M[0, 0] = 2.0
    M = lil_matrix(M)
    retval = [None for _ in range(d)]
    for k in range(d):
        for i in range(n+1):
            ind_hi = min(i + 1, n - 1)
            ind_lo = max(0, i - 1)
            Y[i] = 3 * (pts[k, ind_hi] - pts[k, ind_lo])
        D = linalg.lsqr(M, Y)[0]
        spline_coeffs = np.zeros((4, n))
        spline_coeffs[0, :] = pts[k, :n]  # x₀
        spline_coeffs[1, :] = D[:n]  # x'₀
        spline_coeffs[2, :] = 3 * (pts[k, 1:n + 1] - pts[k, :n]) - 2 * D[:n] - D[1:n + 1]  # -3x₀ + 3x₁ - 2x'₀ - x'₁
        spline_coeffs[3, :] = 2 * (pts[k, :n] - pts[k, 1:n + 1]) + D[:n] + D[1:n + 1]  # 2x₀ - 2x₁ +  x'₀ + x'₁

        retval[k] = spline_coeffs
    return retval


def fit_cubic_spline(pts, open: bool = True):
    if open:
        return _fit_open(pts)


def sample_spline_derivative(spline_coeffs, t: float):
    # here t is generally expected to be t ∈ [0,1]
    return spline_coeffs[1] + t * (2 * spline_coeffs[2] + t * 3 * spline_coeffs[3])


def sample_spline_derivative2(spline_coeffs, t: float):
    # here t is generally expected to be t ∈ [0,1]
    return 2 * spline_coeffs[3] + t * 6 * spline_coeffs[4]


def sample_spline_speed_1(spline_coeffs_x, spline_coeffs_y, t: float):
    dxdt = sample_spline_derivative(spline_coeffs_x, t)
    dydt = sample_spline_derivative(spline_coeffs_y, t)
    return math.hypot(dxdt, dydt)


def sample_spline_speed_2(spline_coeffs_x, spline_coeffs_y, t: float):
    n = spline_coeffs_x.shape[1]
    assert spline_coeffs_x.shape[0] == 4
    assert spline_coeffs_y.shape[0] == 4
    assert n == spline_coeffs_y.shape[1]
    col_ind = max(1, min(int(n), int(math.ceil(t))))
    return sample_spline_speed_1(spline_coeffs_x[:, col_ind - 1], spline_coeffs_y[:, col_ind - 1], t - col_ind + 1)


def calc_curve_length_1(spline_coeffs_x, spline_coeffs_y, n_intervals: int = 100):

    # integrate using Simpson's rule
    # _integrate_simpsons(t->sample_spline_speed(spline_coeffs_x, spline_coeffs_y, t), 0.0, 1.0, n_intervals)

    a = 0.0
    b = 1.0
    n = n_intervals
    h = (b - a) / n
    retval = sample_spline_speed_1(spline_coeffs_x, spline_coeffs_y, a) + sample_spline_speed_1(spline_coeffs_x,
                                                                                                spline_coeffs_y, b)
    flip = True
    for i in range(1, n):
        retval += sample_spline_speed_1(spline_coeffs_x, spline_coeffs_y, a + i * h) * (4 if flip else 2)
        flip = not flip
    return h / 3 * retval


def calc_curve_length_2(spline_coeffs_x, spline_coeffs_y, n_intervals_per_segment: int = 100):
    n = spline_coeffs_x.shape[1]
    assert spline_coeffs_y.shape[1] == n
    assert spline_coeffs_x.shape[0] == 4 and spline_coeffs_y.shape[0] == 4
    len = 0.0
    for i in range(n):
        print("cal curve length: {}/{}".format(i, n))
        len += calc_curve_length_1(spline_coeffs_x[:, i], spline_coeffs_y[:, i], n_intervals=n_intervals_per_segment)
    return len


def calc_curve_param_given_arclen(spline_coeffs_x, spline_coeffs_y, s_arr, max_iterations: int = 50,
                                  curve_length: float = None, epsilon: float = 1e-4, n_intervals_in_arclen: int = 100):
    n_segments = spline_coeffs_x.shape[1]
    assert spline_coeffs_x.shape[0] == 4 and spline_coeffs_y.shape[0] == 4
    assert spline_coeffs_y.shape[1] == n_segments
    n = len(s_arr)
    t_arr = np.zeros((n,))
    s = s_arr[0]
    t = s / curve_length
    if s <= 0.0:
        t = 0.0
    elif s >= curve_length:
        return float(n_segments)
    lo = 0.0
    # print("L: ", curve_length)
    # print("s_max: ", s_arr[-1])
    for (i, s) in tqdm.tqdm(enumerate(s_arr)):
        if s <= 0.0:
            t = 0.0
            t_arr[i] = lo = t
            continue
        elif s >= curve_length:
            t = float(n_segments)
            t_arr[i] = lo = t
            continue
        hi = float(n_segments)
        for iter in range(max_iterations):
            F = arclength_2(spline_coeffs_x, spline_coeffs_y, 0.0, t, n_intervals_in_arclen) - s
            if abs(F) < epsilon:
                break
            DF = sample_spline_speed_2(spline_coeffs_x, spline_coeffs_y, t)
            tCandidate = t - F / DF
            if F > 0:
                hi = t
                t = 0.5 * (lo + hi) if tCandidate <= lo else tCandidate
            else:
                lo = t
                t = 0.5 * (lo + hi) if tCandidate >= hi else tCandidate
        t_arr[i] = lo = t
    return t_arr


def arclength_1(spline_coeffs_x, spline_coeffs_y, t_min: float = 0.0, t_max: float = 1.0, n_intervals: int = 100):
    if math.isclose(t_min, t_max):
        return 0.0

    a = t_min
    b = t_max
    n = n_intervals

    h = (b - a) / n
    retval = sample_spline_speed_1(spline_coeffs_x, spline_coeffs_y, a) + sample_spline_speed_1(spline_coeffs_x,
                                                                                                spline_coeffs_y, b)
    flip = True
    for i in range(1, n):
        retval += sample_spline_speed_1(spline_coeffs_x, spline_coeffs_y, a + i * h) * (4 if flip else 2)
        flip = not flip
    return h / 3 * retval


def arclength_2(spline_coeffs_x, spline_coeffs_y, t_min: float = 0.0, t_max: float = None,
                n_intervals_per_segment: int = 100):
    if t_max is None:
        t_max = spline_coeffs_x.shape[1]
    n = spline_coeffs_x.shape[1]
    assert spline_coeffs_y.shape[1] == n
    assert spline_coeffs_x.shape[0] == 4 and spline_coeffs_y.shape[0] == 4
    if math.isclose(t_min, t_max):
        return 0.0
    len = 0.0
    for i in range(int(math.floor(t_min)), min(int(math.floor(t_max)), n-1) + 1):
        t_lo, t_hi = float(i), i + 1.0

        spline_ind = i + 1
        t_in_min = max(t_lo, t_min) - t_lo
        t_in_max = min(t_hi, t_max) - t_lo

        len += arclength_1(spline_coeffs_x[:, spline_ind - 1], spline_coeffs_y[:, spline_ind - 1], t_in_min, t_in_max,
                           n_intervals_per_segment)

    return len


def sample_spline_1(spline_coeffs, t: float):
    # here t is generally expected to be t ∈ [0,1]
    return spline_coeffs[0] + t * (spline_coeffs[1] + t * (spline_coeffs[2] + t * spline_coeffs[3]))


def sample_spline_2(spline_coeffs, t_arr):
    assert spline_coeffs.shape[0] == 4
    retval = np.zeros((len(t_arr,)))
    for (i, t) in enumerate(t_arr):
        col_ind = max(1, min(spline_coeffs.shape[1], int(math.ceil(t))))
        retval[i] = sample_spline_1(spline_coeffs[:, col_ind - 1], t - col_ind + 1)
    return retval


def sample_spline_theta_1(spline_coeffs_x, spline_coeffs_y, t: float, stepsize = 1e-4):
    t_lo, t_hi = t, t + stepsize
    if t_hi > 1.0:
        t_lo, t_hi = t - min(1000 * stepsize, 0.1), t

    x1 = sample_spline_1(spline_coeffs_x, t_lo)
    x2 = sample_spline_1(spline_coeffs_x, t_hi)
    y1 = sample_spline_1(spline_coeffs_y, t_lo)
    y2 = sample_spline_1(spline_coeffs_y, t_hi)

    # println("(t, lo, hi)  $t   $t_lo   $t_hi, ($(atan(y2-y1, x2-x1)))")

    return math.atan2(y2-y1, x2-x1)


def sample_spline_theta_2(spline_coeffs_x, spline_coeffs_y, t_arr):
    n = spline_coeffs_x.shape[1]
    assert spline_coeffs_y.shape[1] == n
    assert spline_coeffs_x.shape[0] == 4 and spline_coeffs_y.shape[0] == 4
    retval = np.zeros((len(t_arr),))
    for (i, t) in enumerate(t_arr):
        col_ind = max(1, min(n, int(math.ceil(t))))
        retval[i] = sample_spline_theta_1(spline_coeffs_x[:, col_ind - 1], spline_coeffs_y[:, col_ind - 1],
                                          t - col_ind + 1)
    return retval


def sample_spline_curvature_1(spline_coeffs_x, spline_coeffs_y, t: float):
    # computes the signed curvature
    dx = sample_spline_derivative(spline_coeffs_x, t)
    dy = sample_spline_derivative(spline_coeffs_y, t)
    ddx = sample_spline_derivative2(spline_coeffs_x, t)
    ddy = sample_spline_derivative2(spline_coeffs_y, t)
    return (dx*ddy - dy*ddx)/(dx*dx + dy*dy)**1.5


def sample_spline_curvature_2(spline_coeffs_x, spline_coeffs_y, t_arr):
    n = spline_coeffs_x.shape[1]
    assert spline_coeffs_y.shape[1] == n
    assert spline_coeffs_x.shape[0] == 4 and spline_coeffs_y.shape[0] == 4
    retval = np.zeros((len(t_arr),))
    for (i, t) in enumerate(t_arr):
        col_ind = max(1, min(n, int(math.ceil(t))))
        retval[i] = sample_spline_curvature_1(spline_coeffs_x[:, col_ind - 1], spline_coeffs_y[:, col_ind - 1],
                                              t - col_ind + 1)
    return retval


def sample_spline_derivative_of_curvature_1(spline_coeffs_x, spline_coeffs_y, t, stepsize = 1e-4):
    # computes the derivative of the signed curvature
    t_lo, t_hi = t, t + stepsize
    if t_hi > 1.0:
        t_lo, t_hi = t - stepsize, t
    k_hi = sample_spline_curvature_1(spline_coeffs_x, spline_coeffs_y, t_hi)
    k_lo = sample_spline_curvature_1(spline_coeffs_x, spline_coeffs_y, t_lo)
    return (k_hi - k_lo) / stepsize


def sample_spline_derivative_of_curvature_2(spline_coeffs_x, spline_coeffs_y, t_arr, stepsize = 1e-4):
    n = spline_coeffs_x.shape[1]
    assert spline_coeffs_y.shape[1] == n
    assert spline_coeffs_x.shape[0] == 4 and spline_coeffs_y.shape[0] == 4
    retval = np.zeros((len(t_arr),))
    for (i, t) in enumerate(t_arr):
        col_ind = max(1, min(n, int(math.ceil(t))))
        retval[i] = sample_spline_derivative_of_curvature_1(spline_coeffs_x[:, col_ind - 1], spline_coeffs_y[:, col_ind - 1],
                                                            t - col_ind + 1, stepsize = stepsize)
    return retval
