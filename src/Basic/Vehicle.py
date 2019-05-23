from Vec import VecSE2
from Roadway import roadway, utils

"""
Frenet
______
roadind: road index
s: distance along lane
t: lane offset, positive is to left. zero point is the centerline of the lane.
ϕ: lane relative heading
"""


class Frenet:
    def __init__(self, posG: VecSE2.VecSE2, roadWay: roadway.Roadway):
        # roadind: roadway.RoadIndex, s: float, t: float, phi: float
        roadproj = roadway.proj_2(posG, roadWay)
        roadind = roadway.RoadIndex(roadproj.curveproj.ind, roadproj.tag)
        s = roadWay.get_by_roadindex(roadind).s
        t = roadproj.curveproj.t
        phi = utils.mod2pi2(roadproj.curveproj.phi)
        self.roadind = roadind
        self.s = s
        self.t = t
        self.phi = phi


class AgentClass:
    MOTORCYCLE = 1
    CAR = 2
    TRUCK = 3
    PEDESTRIAN = 4


"""
    Vehicle definition which contains a class and a bounding box.
"""


class VehicleDef:
    def __init__(self, class_: int = AgentClass.CAR, length_: int = 4.0, width_: int = 1.8):
        self.class_ = class_
        self.length_ = length_
        self.width_ = width_

    def write(self, fp):
        fp.write("%d %.16e %.16e" % (self.class_, self.length_, self.width_))


NULL_VEHICLEDEF = VehicleDef(AgentClass.CAR, None, None)


class VehicleState:
    def __init__(self, posG: VecSE2.VecSE2, roadWay: roadway.Roadway, v: float):
        self.posG = posG
        self.posF = Frenet(posG, roadWay)
        self.v = v

    def write(self, fp):
        fp.write("%.16e %.16e %.16e" % (self.posG.x, self.posG.y, self.posG.theta))
        fp.write(" %d %.16e %d %d" % (self.posF.roadind.ind.i, self.posF.roadind.ind.t,
                                      self.posF.roadind.tag.segment, self.posF.roadind.tag.lane))
        fp.write(" %.16e %.16e %.16e" % (self.posF.s, self.posF.t, self.posF.phi))
        fp.write(" %.16e" % self.v)

