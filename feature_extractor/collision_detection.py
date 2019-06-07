import math
from src.Record.frame import Frame
from src.Vec import VecE2, VecSE2
from src.Basic.Vehicle import Vehicle
from src.Vec.geom.line_segment import LineSegment, get_polar_angle
from src.Vec.geom.projectile import Projectile, get_intersection_time
from src.Vec.geom.geom import sign, cross_product


class ConvexPolygon:
    def __init__(self, npts):
        self.pts = [None for i in range(npts)]
        self.npts = 0

    def __len__(self):
        return self.npts

    def set(self, poly):
        self.pts = poly.pts
        self.npts = poly.npts


class CollisionCheckResult:
    def __init__(self, is_colliding: bool, A: int, B: int):
        '''

        CollisionCheckResult
        A type to store the result of a collision checker
        # Fields
        - `is_colliding::Bool`
        - `A::Int64` # index of 1st vehicle
        - `B::Int64` # index of 2nd vehicle
        '''

        self.is_colliding = is_colliding
        self.A = A
        self.B = B


class CPAMemory:
    def __init__(self, vehA: ConvexPolygon = ConvexPolygon(4), vehB: ConvexPolygon = ConvexPolygon(4),
                 mink: ConvexPolygon = ConvexPolygon(8)):
        '''

        :param vehA: bounding box for vehicle A
        :param vehB: bounding box for vehicle B
        :param mink: minkowski bounding box
        '''

        self.vehA = vehA
        self.vehB = vehB
        self.mink = mink


def empty(poly: ConvexPolygon):
    poly.npts = 0
    return poly


def isempty(poly: ConvexPolygon):
    return poly.npts == 0


def push(poly: ConvexPolygon, v: VecE2.VecE2):
    poly.pts[poly.npts] = v
    poly.npts += 1

    return poly


def get_signed_area(pts: []):
    npts = len(pts)

    # https://en.wikipedia.org/wiki/Shoelace_formula
    # sign of -1 means clockwise, sign of 1 means counterclockwise

    retval = pts[npts-1].x*pts[0].y - pts[0].x*pts[npts-1].y
    for i in range(npts-1):
        retval += pts[i].x * pts[i+1].y
        retval -= pts[i+1].x * pts[i].y

    return retval/2


def _bounding_radius(veh: Vehicle):
    return math.sqrt(veh.definition.length_ * veh.definition.length_ / 4 + veh.definition.width_ * veh.definition.width_ / 4)


def is_potentially_colliding(A: Vehicle, B: Vehicle):
    vec = A.state.posG - B.state.posG
    delta_square = VecE2.normsquared(VecE2.VecE2(vec.x, vec.y))
    r_a = _bounding_radius(A)
    r_b = _bounding_radius(B)
    return delta_square <= r_a*r_a + 2*r_a*r_b + r_b*r_b


def cyclic_shift_left(arr: list, d: int, n: int):
    for i in range(math.gcd(d, n)):
        # move i-th values of blocks

        temp = arr[i]
        j = i
        while True:
            k = j + d
            if k >= n:
                k = k - n
            if k == i:
                break
            arr[j] = arr[k]
            j = k
        arr[j] = temp
    return arr


def ensure_pts_sorted_by_min_polar_angle(poly: ConvexPolygon):
    npts = poly.npts
    assert npts >= 3
    assert get_signed_area(poly.pts) > 0 # must be counter-clockwise

    # ensure that edges are sorted by minimum polar angle in [0,2π]

    angle_start = math.inf
    index_start = -1

    for i in range(npts):
        seg = get_edge(poly.pts, i, npts)
        theta = math.atan2(seg.B.y - seg.A.y, seg.B.x - seg.A.x)

        if theta < 0:
            theta += 2*math.pi
        if theta < angle_start:
            angle_start = theta
            index_start = i
    if index_start != 0:
        poly.pts = cyclic_shift_left(poly.pts, index_start, npts)
    return poly


def to_oriented_bounding_box_1(retval: ConvexPolygon, center: VecSE2.VecSE2, len: float, wid: float):

    assert len > 0
    assert wid > 0
    assert center.theta is not None
    assert center.x is not None
    assert center.y is not None

    x = VecE2.polar(len/2, center.theta)
    y = VecE2.polar(wid/2, center.theta + math.pi/2)

    C = VecSE2.convert(center)
    retval.pts[0] = x - y + C
    retval.pts[1] = x + y + C
    retval.pts[2] = -x + y + C
    retval.pts[3] = -x - y + C
    retval.npts = 4

    retval.set(ensure_pts_sorted_by_min_polar_angle(retval))

    return retval


def to_oriented_bounding_box_2(retval: ConvexPolygon, veh: Vehicle):
    return to_oriented_bounding_box_1(retval, veh.get_center, veh.definition.length_, veh.definition.width_)


def get_edge(pts: list, i: int, npts: int):
    a = pts[i]
    if i + 1 < npts:
        b = pts[i + 1]
    else:
        b = pts[0]
    return LineSegment(a, b)


def get_poly_edge(poly: ConvexPolygon, i: int):
    return get_edge(poly.pts, i, poly.npts)


def get_collision_time(ray: VecSE2.VecSE2, poly: ConvexPolygon, ray_speed: float):
    min_col_time = math.inf
    for i in range(len(poly)):
        seg = get_poly_edge(poly, i)
        col_time = get_intersection_time(Projectile(ray, ray_speed), seg)
        if col_time and col_time < min_col_time:
            min_col_time = col_time
    return min_col_time


def mirror(poly: ConvexPolygon):
    for i in range(len(poly)):
        poly.pts[i] = -poly.pts[i]

    poly.set(ensure_pts_sorted_by_min_polar_angle(poly))

    return poly


def minkowksi_sum(retval: ConvexPolygon, P: ConvexPolygon, Q: ConvexPolygon):
    '''

    For two convex polygons P and Q in the plane with m and n vertices, their Minkowski sum is a
    convex polygon with at most m + n vertices and may be computed in time O (m + n) by a very simple procedure,
    which may be informally described as follows.
    Assume that the edges of a polygon are given and the direction, say, counterclockwise, along the polygon boundary.
    Then it is easily seen that these edges of the convex polygon are ordered by polar angle.
    Let us merge the ordered sequences of the directed edges from P and Q into a single ordered sequence S.
    Imagine that these edges are solid arrows which can be moved freely while keeping them parallel to their original direction.
    Assemble these arrows in the order of the sequence S by attaching the tail of the next arrow to the head of the previous arrow.
    It turns out that the resulting polygonal chain will in fact be a convex polygon which is the Minkowski sum of P and Q.

    '''

    retval.set(empty(retval))

    index_P = 0
    index_Q = 0

    theta_p = get_polar_angle(get_edge(P.pts, index_P, P.npts))
    theta_q = get_polar_angle(get_edge(Q.pts, index_Q, Q.npts))

    while index_P < len(P) or index_Q < len(Q):
        # select next edge with minimum polar angle

        if theta_p == theta_q:
            seg_p = get_edge(P.pts, index_P, P.npts)
            seg_q = get_edge(Q.pts, index_Q, Q.npts)

            O = (P.pts[0] + Q.pts[0]) if isempty(retval) else retval.pts[retval.npts - 1]
            retval.set(push(retval, O + seg_p.B - seg_p.A + seg_q.B - seg_q.A))
            index_P += 1
            theta_p = get_polar_angle(get_edge(P.pts, index_P, P.npts)) if index_P < len(P) else math.inf
            index_Q += 1
            theta_q = get_polar_angle(get_edge(Q.pts, index_Q, Q.npts)) if index_Q < len(Q) else math.inf
        elif theta_p <= theta_q:
            seg = get_edge(P.pts, index_P, P.npts)
            O = (P.pts[0] + Q.pts[0]) if isempty(retval) else retval.pts[retval.npts - 1]
            retval.set(push(retval, O + seg.B - seg.A))
            index_P += 1
            theta_p = get_polar_angle(get_edge(P.pts, index_P, P.npts)) if index_P < len(P) else math.inf
        else:
            seg = get_edge(Q.pts, index_Q, Q.npts)
            O = (P.pts[0] + Q.pts[0]) if isempty(retval) else retval.pts[retval.npts - 1]
            retval.set(push(retval, O + seg.B - seg.A))
            index_Q += 1
            theta_q = get_polar_angle(get_edge(Q.pts, index_Q, Q.npts)) if index_Q < len(Q) else math.inf

    retval.set(ensure_pts_sorted_by_min_polar_angle(retval))

    return retval


def minkowski_difference(retval: ConvexPolygon, P: ConvexPolygon, Q: ConvexPolygon):

    Q.set(mirror(Q))
    retval.set(minkowksi_sum(retval, P, Q))

    return retval


def is_colliding_1(mem: CPAMemory):
    return is_colliding_2(mem.vehA, mem.vehB, mem.mink)


def is_colliding_2(P: ConvexPolygon, Q: ConvexPolygon, temp: ConvexPolygon):
    temp.set(minkowski_difference(temp, P, Q))
    return in_poly(VecE2.VecE2(0, 0), temp)


def in_poly(v: VecE2.VecE2, poly: ConvexPolygon):
    previous_side = 0

    for i in range(len(poly)):
        seg = get_edge(poly.pts, i, poly.npts)
        affine_segment = seg.B - seg.A
        affine_point = v - seg.A
        current_side = int(sign(cross_product(affine_segment, affine_point)))  # sign indicates side
        if current_side == 0:
            # outside or over an edge
            return False
        elif previous_side == 0:
            # first segment
            previous_side = current_side
        elif previous_side != current_side:
            # all most be on the same side
            return False

    return True


def get_first_collision(scene: Frame, target_index: int, mem: CPAMemory = CPAMemory()):
    A = target_index

    vehA = scene[A]
    mem.vehA.set(to_oriented_bounding_box_2(mem.vehA, vehA))

    for B in range(scene.n):
        vehB = scene[B]
        if B != A:
            mem.vehB.set(to_oriented_bounding_box_2(mem.vehB, vehB))
            if is_potentially_colliding(vehA, vehB) and is_colliding_1(mem):
                return CollisionCheckResult(True, A, B)

    return CollisionCheckResult(False, A, 0)


