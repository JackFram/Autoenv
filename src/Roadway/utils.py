import math


def mod2pi2(x: float):
    val = x % (2 * math.pi)
    if val > math.pi:
        val -= 2 * math.pi
    return val
