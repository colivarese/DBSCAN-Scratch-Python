import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from random import randint

class dbscan():
    def __init__(self, eps:float, minPts:int, n_iters:int) -> None:
        self.eps = eps
        self.minPts = minPts
        self.n_iters = n_iters

    def fit(self, data):
        data = np.asarray(data)
        data = self.add_index(data)
        C = dict()
        clusters = list()
        noise = list()
        data = self.mark_as_unvisited(data)
        count = 0
        #while (self.all_visited(data)):
        for _ in range(self.n_iters):
            random_start = np.random.randint(0, len(data),1)[0]
            isVisited = self.is_visited(data, random_start)
            if not isVisited:
                data = self.mark_as_visited(data, random_start)
                distances = self.get_minPts_distance_to_objects(data, random_start)
                C[count] = clusters
                if (self.satisfy_minPts(distances)):
                    #clusters.append(data[random_start,1:3])
                    clusters.append(random_start)
                    for p in distances:
                        
                        p = np.asarray(p)
                        if not self.is_visited(data, p):
                            data = self.mark_as_visited(data, p)
                            p_dist = self.get_minPts_distance_to_objects(data,p)
                            if (self.satisfy_minPts(p_dist)):
                                #distances.append(p_dist)
                                distances = np.concatenate((distances,p_dist))
                                distances = np.asarray(list(set(distances)))
                                a=1
                        if not self.check_in_clusters(p, C):
                        #if int(p) not in clusters:
                            clusters.append(int(p))
                
                else:
                    noise.append(random_start)
                    data = self.mark_as_visited(data, random_start)
            count += 1
            clusters = []
        C = {k:v for k,v in C.items() if v}
        clusters = np.unique(clusters)
        noise = np.unique(noise)
        self.plot_clusters(data, C, noise)

        
    def check_in_clusters(self ,p, clusters):
        for k in clusters:
            c = clusters[k]
            if p in c:
                return True
        return False

    def plot_clusters(self, data, clusters, noise):
        plt.figure()
        colors = []
        n = len(clusters)
        for i in range(n):
            colors.append('#%06X' % randint(0, 0xFFFFFF))
        #colors = ['r','g','b']
        for i, k in enumerate(clusters):
            for p, c in zip(data, clusters[k]):
                x = data[c,1]
                y = data[c,2]
                plt.scatter(x,y, color=colors[i])
            for p, n in zip(data, noise):
                x = data[n,1]
                y = data[n,2]
                plt.scatter(x,y, color='black')
        plt.show()


    def mark_as_unvisited(self,data):
        num_cols = data.shape[0]
        visited = np.zeros((num_cols,1))
        data = np.append(data, visited, axis = 1)
        return data

    def mark_as_visited(self, data, idx):
        data[idx,-1] = 1
        return data

    def get_minPts_distance_to_objects(self, data, idx):
        p = data[idx,1:3][...,None].T
        distances = (cdist(data[:,1:3], p ,'euclidean'))
        eps_neigh = list()
        for idx, d in enumerate(distances):
            if d <= self.eps:
                eps_neigh.append(idx)
        return eps_neigh

    def satisfy_minPts(self, distances):
        return len(distances) >= self.minPts

    def add_index(self, data):
        idx_col = np.array(range(0,len(data)))[...,None]
        data_with_idx = np.append(idx_col, data, 1)
        return data_with_idx

    def is_visited(self, data, idx):
        p = data[idx,:]
        return p[-1] == 1

    def all_visited(self, data):
        tmp = np.min(data[:,-1])
        return tmp == 0
