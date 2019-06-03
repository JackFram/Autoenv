from src.Vec import VecE2
import math
from src.Roadway.utils import mod2pi2


class LineSegment:
    def __init__(self, A: VecE2.VecE2, B: VecE2.VecE2):
        self.A = A
        self.B = B


def get_polar_angle(seg: LineSegment):
    return mod2pi2(math.atan2(seg.B.y - seg.A.y, seg.B.x - seg.A.x))

