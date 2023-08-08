import numpy as np
import time
import scipy.constants as constants
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
from pathlib import Path

from planeWaveClasses import DiscretePlaneWave, TFSFBox


# Basic Model parameters
dimensions = 3

# Main 3D grid parameters
number_x = 25
number_y = 25
number_z = 25
# TF/SF box coordinates 
corners=np.array([[5, 5, 5], [20, 20, 20]], dtype=np.int32, order='C')

# Main Grid Numerber of Iterations
time_duration = 100
snapshot = 10

# Spatial and temporal steps and key parameters for both grids
dx = 0.001
dy = 0.001
dz = 0.001
epsilon_r = 1.0
mu_r = 1.0
dt = 1.0 / (constants.c * np.sqrt((1.0/dx**2)+(1.0/dy**2)+(1.0/dz**2)))

# Source information
ppw = 20*dt

phi = 63.4
dPhi = 2
theta = 36.7
dTheta = 1

angles = np.array([[-np.pi/2, 180+63.4, 2, 180-36.7, 1],
                   [np.pi/2, 63.4, 2, 36.7, 1]])      
# Plane wave E field polarization angle with the direction of propagation 

# DPW information a buffer for avoid boundary effects
number = 50

Path("./snapshots").mkdir(parents=True, exist_ok=True)

start = time.time()

DPW1 = DiscretePlaneWave(time_duration, dimensions, number_x, number_y, number_z)
DPW2 = DiscretePlaneWave(time_duration, dimensions, number_x, number_y, number_z)

SpaceGrid = TFSFBox(number_x, number_y, number_z, corners, time_duration, dimensions, 2)
SpaceGrid.getFields([DPW1, DPW2], snapshot, angles, number, dx, dy, dz, dt, ppw)
        
end = time.time()
print("Elapsed (with compilation) = %s sec" % (end - start))
