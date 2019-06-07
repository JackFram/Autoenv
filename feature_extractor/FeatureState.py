GOOD = 0  # value is perfectly A-okay
INSUF_HIST = 1  # value best-guess was made due to insufficient history (ex, acceleration set to zero due to one timestamp)
MISSING = 2  # value is missing (no car in front, etc.)
CENSORED_HI = 3  # value is past an operating threshold
CENSORED_LO = 4  # value is below an operating threshold


class FeatureValue:
    def __init__(self, v: float, i: int = GOOD):
        self.v = v
        self.i = i
        

def inverse_ttc_to_ttc(inv_ttc: FeatureValue, censor_hi: float = 30.0):
    if inv_ttc.i == MISSING:
        # if the value is missing then censor hi and set missing
        return FeatureValue(censor_hi, MISSING)
    elif inv_ttc.i == GOOD and inv_ttc.v == 0.0:
        # if the car in front is pulling away, then set to a censored hi value
        return FeatureValue(censor_hi, CENSORED_HI)
    else:
        # even if the value was censored hi, can still take the inverse
        ttc = 1.0 / inv_ttc.v
        if ttc > censor_hi:
            return FeatureValue(censor_hi, CENSORED_HI)
        else:
            return FeatureValue(ttc, GOOD)