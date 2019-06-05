class AccelTurnrate:
    def __init__(self, a: float, omega: float):
        self.a = a
        self.omega = omega

def propagate(veh, action, roadway, delta_t):

    raise NotImplementedError

    # L = veh.def.a + veh.def.b
    # l = -veh.def.b
    #
    # a = action.a # accel [m/s²]
    # δ = action.δ # steering wheel angle [rad]
    #
    # x = veh.state.posG.x
    # y = veh.state.posG.y
    # θ = veh.state.posG.θ
    # v = veh.state.v
    #
    # s = v*Δt + a*Δt*Δt/2 # distance covered
    # v′ = v + a*Δt
    #
    # if abs(δ) < 0.01 # just drive straight
    #     posG = veh.state.posG + polar(s, θ)
    # else # drive in circle
    #
    #     R = L/tan(δ) # turn radius
    #
    #     β = s/R
    #     xc = x - R*sin(θ) + l*cos(θ)
    #     yc = y + R*cos(θ) + l*sin(θ)
    #
    #     θ′ = mod(θ+β, 2π)
    #     x′ = xc + R*sin(θ+β) - l*cos(θ′)
    #     y′ = yc - R*cos(θ+β) - l*sin(θ′)
    #
    #     posG = VecSE2(x′, y′, θ′)
    #
    # VehicleState(posG, roadway, v′)

