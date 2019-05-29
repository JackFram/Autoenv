# Zhihao Zhang
# NGSIM dataset processor trajectory_smoothing.py file

import math
import numpy as np
from src.MvNormal import MvNormal


class VehicleSystem:
    def __init__(self, process_noise: float = 0.077, observation_noise: float = 16.7,
                 control_noise_accel: float = 16.7, control_noise_turnrate: float = 0.46):
        delta_t = 0.1
        H = np.array([[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0]])
        r = process_noise
        R = MvNormal.MvNormal(np.array([0, 0, 0, 0]), np.diag([r * 0.01, r * 0.01, r * 0.00001, r * 0.1]))  # process, TODO: tune this
        q = observation_noise
        Q = MvNormal.MvNormal(np.array([0, 0]), np.diag([q, q]))  # obs, TODO: tune this

        n_integration_steps = 10

        self.H = H
        self.R = R
        self.Q = Q
        self.delta_t = delta_t
        self.n_integration_steps = n_integration_steps
        self.control_noise_accel = control_noise_accel
        self.control_noise_turnrate = control_noise_turnrate


def draw_proc_noise(v: VehicleSystem):
    return v.R.rand_sample()


def draw_obs_noise(v: VehicleSystem):
    return v.Q.rand_sample()


def get_process_noise_covariance(v: VehicleSystem):
    return v.R.cov


def get_observation_noise_covariance(v: VehicleSystem):
    return v.Q.cov


"""
    To derive the covariance of the additional motion noise,
    we first determine the covariance matrix of the noise in control space
    Inputs:
        - ν is the vehicle concrete type
        - u is the control [a,ω]ᵀ
"""


def get_control_noise_in_control_space(v: VehicleSystem, u: list):
    return np.array([[v.control_noise_accel, 0.0],
                    [0.0, v.control_noise_turnrate]])

"""
    To derive the covariance of the additional motion noise,
    we transform the covariance of noise in the control space
    by a linear approximation to the derivative of the motion function
    with respect to the motion parameters
    Inputs:
        - ν is the vehicle concrete type
        - u is the control [a,ω]ᵀ
        - x is the state estimate [x,y,θ,v]ᵀ
"""


def get_transform_control_noise_to_state_space(v: VehicleSystem, u: list, x: list):

    _, _, theta, v_ = x[0], x[1], x[2], x[3]
    _, omega = u[0], u[1]
    omega_square = omega * omega
    delta_t = v.delta_t

    phi = theta + omega * delta_t

    if abs(omega) < 1e-6:
        return np.array([[0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, delta_t],
                        [delta_t, 0.0]])
    else:
        p = v_/omega_square*(math.sin(theta) - math.cos(phi)) + v_/omega*math.cos(phi)*delta_t
        q = -v_/omega_square*(math.sin(theta) - math.cos(phi)) + v_/omega*math.cos(phi)*delta_t
        return np.array([[0.0, p],
                        [0.0, q],
                        [0.0, delta_t],
                        [delta_t, 0.0]])


"""
    Vehicle dynamics, return the new state
    Inputs:
        - ν is the vehicle concrete type
        - x is the state estimate [x,y,θ,v]ᵀ
        - u is the control [a,ω]ᵀ
"""


def step(v: VehicleSystem, x: list, u: list):

    a_, omega = u[0], u[1]
    x_, y_, theta, v_ = x[0], x[1], x[2], x[3]

    delta_t = v.delta_t/v.n_integration_steps

    for i in range(v.n_integration_steps):

        if abs(omega) < 1e-6: # simulate straight
            x_ += v_*math.cos(theta) * delta_t
            y_ += v_*math.sin(theta) * delta_t
        else: # simulate with an arc
            x_ += (v_/omega)*(math.sin(theta + omega*delta_t) - math.sin(theta))
            y_ += (v_/omega)*(math.cos(theta) - math.cos(theta + omega*delta_t))

        theta += omega * delta_t
        v_ += a_ * delta_t

    return np.array([x_, y_, theta, v_])


"""
    Vehicle observation, returns a saturated observation
    Inputs:
        - ν is the vehicle concrete type
        - x is the state estimate [x,y,θ,v]ᵀ
"""


def observe(v: VehicleSystem, x: list):
    return np.array([x[0], x[1]])


"""
    Computes the observation Jacobian (H matrix)
    Inputs:
        - ν is the vehicle concrete type
        - x is the state estimate [x,y,θ,v]ᵀ
"""


def compute_observation_jacobian(v: VehicleSystem, x: list):
    return v.H

"""
    Computes the dynamics Jacobian
    Inputs:
        - ν is the vehicle
        - x is the vehicle state [x,y,θ,v]ᵀ
        - u is the control [a,ω]
"""


def compute_dynamics_jacobian(v: VehicleSystem, x: list, u: list):
    delta_t = v.delta_t
    theta, v_ = x[2], x[3]
    omega = u[1]
    if abs(omega) < 1e-6:
        # drive straight
        return np.array([[1.0, 0.0, -v_*math.sin(theta)*delta_t, math.cos(theta)*delta_t],
                        [0.0, 1.0, v_*math.cos(theta)*delta_t, math.sin(theta)*delta_t],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])

    else:
        # drive in an arc
        phi = theta + omega * delta_t
        return np.array([[1.0, 0.0, v_/omega*(-math.cos(theta) + math.cos(phi)), -1/omega*math.sin(theta) + 1/omega*math.sin(phi)],
                        [0.0, 1.0, v_/omega*(-math.sin(theta) + math.sin(phi)), 1/omega*math.cos(theta) - 1/omega*math.cos(phi)],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])


def EKF(v: VehicleSystem, mu: list, Cov, u: list, o: list):
    """

    :param v: VehicleSystem
    :param mu: mean of belief at time t-1
    :param Cov: cov of belief at time t-1
    :param u: next applied control
    :param o: observation for time t
    :return: mu_next np.array, Cov_next np.array
    """
    G = compute_dynamics_jacobian(v, mu, u)
    mu_bar = step(v, mu, u)
    R = get_process_noise_covariance(v)
    M = get_control_noise_in_control_space(v, u)
    V = get_transform_control_noise_to_state_space(v, u, list(mu_bar))
    Cov_bar = np.matmul(np.matmul(G, Cov), np.transpose(G)) + R + np.matmul(np.matmul(V, M), np.transpose(V))
    H = compute_observation_jacobian(v, list(mu_bar))
    A = np.matmul(Cov_bar, np.transpose(H))
    B = (np.matmul(np.matmul(H, Cov_bar), np.transpose(H)) + get_observation_noise_covariance(v))

    K = np.linalg.lstsq(B.transpose(), A.transpose())[0].transpose()
    mu_next = mu_bar + np.matmul(K, (o - observe(v, list(mu_bar))))
    Cov_next = Cov_bar - np.matmul(K, np.matmul(H, Cov_bar))

    return mu_next, Cov_next


class SimulationResults:
    def __init__(self, x_arr, z_arr, u_arr, mu_arr, Cov_arr):
        self.x_arr = x_arr
        self.z_arr = z_arr
        self.u_arr = u_arr
        self.mu_arr = mu_arr
        self.Cov_arr = Cov_arr

