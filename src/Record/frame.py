class Frame:
    def __init__(self):
        '''
        entities::List{Vehicle} , Scene
        n::Int
        '''
        self.entities = []
        self.n = 0

    def __getitem__(self, item):
        return self.entities[item]

    def findfirst(self, id: int):
        for entity_index in range(self.n):
            entity = self.entities[entity_index]
            if entity.id == id:
                return entity_index

        return None

    def init(self, n: int):
        for i in range(n):
            self.entities.append(None)


