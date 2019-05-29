class MultiFeatureExtractor:
    def __init__(self, extractors: []):
        '''

        :param extractors: List{FeatureExtractors}
        '''
        self.extractors = extractors
        self.lengths = [len(subext) for subext in extractors]
        self.num_features = sum(self.lengths)
        self.features = [0 for i in range(self.num_features)]

