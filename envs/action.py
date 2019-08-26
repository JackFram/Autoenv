from src.Basic.Vehicle import Vehicle, VehicleState
from src.Roadway.roadway import Roadway
import math
from src.Vec.VecSE2 import VecSE2


class AccelTurnrate:
    def __init__(self, a: float, omega: float):
        '''

        :param a: acceleration
        :param omega: turning rate
        '''
        self.a = a
        self.omega = omega


def propagate(veh: Vehicle, action: AccelTurnrate, roadway: Roadway, delta_t: float,
              n_integration_steps: int = 4):
    '''
    propagate the vehicle state according to the specific action
    :param veh: the vehicle that will be propagated
    :param action: the action (acceleration, turning rate)
    :param roadway: the roadway information
    :param delta_t: how long our action will last
    :param n_integration_steps: the integration interval number
    :return: the propagated vehicle state
    '''
    a = action.a  # accel

    omega = action.omega  # turnrate

    x = veh.state.posG.x
    y = veh.state.posG.y
    theta = veh.state.posG.theta
    v = veh.state.v

    sigma_t = delta_t / n_integration_steps

    for i in range(n_integration_steps):
        x += v * math.cos(theta) * sigma_t
        y += v * math.sin(theta) * sigma_t
        theta += omega * sigma_t
        v += a * sigma_t
        # print("step {},  x: {}, y: {}, theta: {}, v: {}".format(i+1, x, y, theta, v))

    posG = VecSE2(x, y, theta)

    retval = VehicleState()
    retval.set(posG, roadway, v)

    return retval
