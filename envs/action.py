from src.Basic.Vehicle import Vehicle, VehicleState, Frenet
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
    :param veh:
    :param action:
    :param roadway:
    :param delta_t:
    :param n_integration_steps:
    :return:
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

    posG = VecSE2(x, y, theta)

    retval = VehicleState()
    retval.set(posG, roadway, v)

    return retval
