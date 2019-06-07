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

    def deleteat(self, entity_index: int):
        for i in range(entity_index, self.n - 1):
            self.entities[i] = self.entities[i + 1]
        self.n -= 1

    def delete_by_id(self, id: int):
        entity_index = self.findfirst(id)
        if entity_index is not None:
            self.deleteat(entity_index)

    def init(self, n: int):
        for i in range(n):
            self.entities.append(None)

    def push(self, entity):
        self.entities[self.n] = entity
        self.n += 1

    def empty(self):
        self.n = 0

    def __setitem__(self, key, value):
        self.entities[key] = value


def copyto(dest: Frame, src: Frame):
    for i in range(src.n):
        dest.entities[i] = src.entities[i]
    dest.n = src.n
    return dest


