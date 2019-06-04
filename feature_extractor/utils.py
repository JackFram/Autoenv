from feature_extractor.CarLidarFeatureExtractor import CarLidarFeatureExtractor
from feature_extractor.CoreFeatureExtractor import CoreFeatureExtractor
from feature_extractor.ForeForeFeatureExtractor import ForeForeFeatureExtractor
from feature_extractor.TemporalFeatureExtractor import TemporalFeatureExtractor
from feature_extractor.WellBehavedFeatureExtractor import WellBehavedFeatureExtractor
from feature_extractor.feature_extractor import MultiFeatureExtractor
from feature_extractor.interface import FeatureValue
from feature_extractor import FeatureState


def build_feature_extractor(params={}):
    subexts = list()
    subexts.append(CoreFeatureExtractor())
    subexts.append(TemporalFeatureExtractor())
    subexts.append(WellBehavedFeatureExtractor())
    subexts.append(CarLidarFeatureExtractor(20, carlidar_max_range=50.))
    subexts.append(ForeForeFeatureExtractor())
    ext = MultiFeatureExtractor(subexts)
    return ext


def inverse_ttc_to_ttc(inv_ttc: FeatureValue, censor_hi: float = 30.0):
    if inv_ttc.i == FeatureState.MISSING:
        # if the value is missing then censor hi and set missing
        return FeatureValue(censor_hi, FeatureState.MISSING)
    elif inv_ttc.i == FeatureState.GOOD and inv_ttc.v == 0.0:
        # if the car in front is pulling away, then set to a censored hi value
        return FeatureValue(censor_hi, FeatureState.CENSORED_HI)
    else:
        # even if the value was censored hi, can still take the inverse
        ttc = 1.0 / inv_ttc.v
        if ttc > censor_hi:
            return FeatureValue(censor_hi, FeatureState.CENSORED_HI)
        else:
            return FeatureValue(ttc, FeatureState.GOOD)


