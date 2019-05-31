class Frame:
    def __init__(self):
        '''
        entities::List{Vehicle}
        n::Int
        '''
        self.entities = []
        self.n = 0

    def __getitem__(self, item):
        return self.entities[item]

