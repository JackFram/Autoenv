from src.Vec import VecSE2


class Projectile:
    def __init__(self,pos: VecSE2.VecSE2, v: float):
        self.pos = pos
        self.v = v

