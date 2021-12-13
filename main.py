import numpy as np
import math
import matplotlib.pyplot as plt
from DBSCAN import dbscan
np.random.seed(42)


# Function for creating datapoints in the form of a circle
def PointsInCircum(r,n=100):
    return [(math.cos(2*math.pi/n*x)*r+np.random.normal(-30,30),math.sin(2*math.pi/n*x)*r+np.random.normal(-30,30)) for x in range(1,n+1)]

data = np.asarray(PointsInCircum(400,700))

DBSCAN = dbscan(20, 4, 100)

DBSCAN.fit(data)