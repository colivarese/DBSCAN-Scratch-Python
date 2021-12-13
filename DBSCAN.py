import numpy as np
from scipy.spatial.distance import cdist

class dbscan():
    def __init__(self, eps:float, minPts:int, n_iters:int) -> None:
        self.eps = eps
        self.minPts = minPts
        self.n_iters = n_iters

    def fit(self, data):
        clusters = list()
        data = self.mark_as_unvisited(data)
        random_start = np.random.randint(0, len(data),1)
        data = self.mark_as_visited(data, random_start)
        distances = self.get_minPts_distance_to_objects(data, random_start)
        if (self.satisfy_minPts(distances)):
            clusters.append(data[random_start,0:2])
            for n in distances:
                return
        for _ in range(self.n_iters):
            return

    def mark_as_unvisited(self,data):
        num_cols = data.shape[0]
        visited = np.zeros((num_cols,1))
        data = np.append(data, visited, axis = 1)
        return data

    def mark_as_visited(self, data, idx):
        data[idx,-1] = 1
        return data

    def get_minPts_distance_to_objects(self, data, idx):
        distances = (cdist(data[:,0:2], data[idx,0:2] ,'euclidean'))
        eps_neigh = list()
        for d in distances:
            if d <= self.eps:
                eps_neigh.append(d)
            else:
                return eps_neigh

    def satisfy_minPts(self, distances):
        return len(distances) >= self.minPts
