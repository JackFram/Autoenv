from feature_extractor.CarLidarFeatureExtractor import CarLidarFeatureExtractor
from feature_extractor.CoreFeatureExtractor import CoreFeatureExtractor
from feature_extractor.ForeForeFeatureExtractor import ForeForeFeatureExtractor
from feature_extractor.TemporalFeatureExtractor import TemporalFeatureExtractor
from feature_extractor.WellBehavedFeatureExtractor import WellBehavedFeatureExtractor
from feature_extractor.feature_extractor import MultiFeatureExtractor
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


