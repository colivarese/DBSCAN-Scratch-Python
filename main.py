import numpy as np
import math
import matplotlib.pyplot as plt
from DBSCAN import dbscan
np.random.seed(42)


# Function for creating datapoints in the form of a circle
def PointsInCircum(r,n=100):
    return [(math.cos(2*math.pi/n*x)*r+np.random.normal(-30,30),math.sin(2*math.pi/n*x)*r+np.random.normal(-30,30)) for x in range(1,n+1)]

data_1 = np.asarray(PointsInCircum(200,700))
#data_2 = np.asarray(PointsInCircum(800,700))
#data = np.concatenate((data_1, data_2))


DBSCAN = dbscan(200, 100, 200)

DBSCAN.fit(data_1)

#plt.figure()
#plt.scatter(data[:,0], data[:,1])
#plt.show()