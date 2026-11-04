from shapely.geometry import LinearRing
class Obstacle:
    def __init__(self, actor):
        self.actor_id = actor.id
        self.actor_type = actor.type_id
        self.actor_transform = actor.get_transform()
        world_vertices = actor.bounding_box.get_world_vertices(self.actor_transform)
        vertexes = [(vertex.x, vertex.y) for vertex in world_vertices]
        coordinates = [vertexes[0], vertexes[2], vertexes[6], vertexes[4]]
        self.actor_linear_ring = LinearRing(coordinates)
    
    def get_actor_linear_ring(self):
        return self.actor_linear_ring
    
    def get_actor_transform(self):
        return self.actor_transform
    
    def get_actor_type(self):
        return self.actor_type
    
    def get_actor_id(self):
        return self.actor_id