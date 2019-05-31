from src.Vec import VecSE2, VecE2
from src.Vec.geom.line_segment import LineSegment
from src.Vec.geom import geom
import math


class Projectile:
    def __init__(self,pos: VecSE2.VecSE2, v: float):
        self.pos = pos
        self.v = v


def get_intersection_time(A: Projectile, seg: LineSegment):
    o = VecE2.VecE2(A.pos.x, A.pos.y)
    v_1 = o - seg.A
    v_2 = seg.B - seg.A
    v_3 = VecE2.polar(1.0, A.pos.theta + math.pi/2)

    denom = geom.dot_product(v_2,v_3)

    if not math.isclose(denom, 0.0, abs_tol=1e-10):
        d_1 = geom.cross_product(v_2, v_1) / denom  # time for projectile (0 ≤ t₁)
        t_2 = geom.dot_product(v_1, v_3) / denom  # time for segment (0 ≤ t₂ ≤ 1)
        if 0.0 <= d_1 and 0.0 <= t_2 <= 1.0:
            return d_1 / A.v
    else:
        # denom is zero if the segment and the projectile are parallel
        # only collide if they are perfectly aligned
        if geom.are_collinear(A.pos, seg.A, seg.B):
            dist_a = VecE2.normsquared(seg.A - o)
            dist_b = VecE2.normsquared(seg.B - o)
            return math.sqrt(min(dist_a, dist_b)) / A.v

    return math.inf

