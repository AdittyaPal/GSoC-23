import numpy as np
import scipy.constants as constants
from planeWaveModules import getIntegerForAngles, getProjections, getGridFields

class DiscretePlaneWave():
    '''
    Class to implement the discrete plane wave (DPW) formulation as described in
    Tan, T.; Potter, M. (2010). 
    FDTD Discrete Planewave (FDTD-DPW) Formulation for a Perfectly Matched Source in TFSF Simulations. ,
    58(8), 0–2648. doi:10.1109/tap.2010.2050446 
    __________________________
    
    Instance variables:
    --------------------------
        m, int array           : stores the integer mappings, m_x, m_y, m_z which determine the rational angles
                                 last element stores max(m_x, m_y, m_z)
        directions, int array  : stores the directions of propagation of the DPW
        dimensions, int        : stores the number of dimensions in which the simulation is run (2D or 3D)
        time_dimension, int    : stores the time length over which the simulation is run
        E_fileds, double array : stores the electric flieds associated with the 1D DPW
        H_fields, double array : stores the magnetic fields associated with the 1D DPW
    '''
    
    
    def __init__(self, time_dimension, dimensions, n_x, n_y, n_z):     
        '''
        Defines the instance variables of class DiscretePlaneWave()
        __________________________

        Input parameters:
        --------------------------
            time_dimension, int : local variable to store the time length over which the simulation is run 
            dimensions, int     : local variable to store the number of dimensions in which the simulation is run
        '''
        self.m = np.zeros(dimensions+1, dtype=np.int32)          #+1 to store the max(m_x, m_y, m_z)
        self.directions = np.zeros(dimensions, dtype=np.int32)
        self.dimensions = dimensions
        self.time_dimension = time_dimension
        self.length = 0
        self.projections = np.zeros(dimensions)
        self.ds = 0
        self.E_fields = []   
        self.H_fields = []
        
    def initializeGrid(self, dl, dt):
        '''
        Method to initialize the one dimensions grids for the DPW
        __________________________

        Input parameters:
        --------------------------
            length_dimension, int : stores the spatial length of the ggrids for the DPW
            dl, double            : stores the spatial seperation between two adjacent elements of the DPW array
            dt, double            : stores the temporal separation between two adjacent rows of the DPW array
        __________________________

        Returns:
        --------------------------
            E_fields, double array :       stores the electric field values for the DPW
                                           first index denotes the spatial dimension
                                           second index denotes the spatial position (grid cell position index)
                                           thid index denotes time (grid cell time index)
            H_fields, double array :       stores the magnetic field values for the DPW
                                           first index denotes the spatial dimension
                                           second index denotes the spatial position (grid cell position index)
                                           thid index denotes time (grid cell time index)
            E_coefficients, double array : stores the coefficients of the fields in the equation to update electric fields
            H_coefficients, double array : stores the coefficients of the fields in the equation to update magnetic fields

        '''
        self.E_fields = np.zeros((self.dimensions, self.length), order='C')
        self.H_fields = np.zeros((self.dimensions, self.length), order='C')
    
        E_coefficients = np.zeros(3*self.dimensions)     #coefficients in the update equations of the electric field
        H_coefficients = np.zeros(3*self.dimensions)     #coefficients in the update equations of the magnetic field
        impedance = constants.c*constants.mu_0   #calculate the impedance of free space 
    
        for i in range(self.dimensions): #loop to calculate the coefficients for each dimension
            E_coefficients[3*i] = 1.0
            E_coefficients[3*i+1] = dt/(constants.epsilon_0*dl[(i+1)%self.dimensions])
            E_coefficients[3*i+2] = dt/(constants.epsilon_0*dl[(i+2)%self.dimensions])        
            
            H_coefficients[3*i] = 1.0
            H_coefficients[3*i+1] = dt/(constants.mu_0*dl[(i+2)%self.dimensions])
            H_coefficients[3*i+2] = dt/(constants.mu_0*dl[(i+1)%self.dimensions])
        
        return E_coefficients, H_coefficients
    
    def runDiscretePlaneWave(self, psi, phi, Delta_phi, theta, Delta_theta, number, dx, dy, dz):
        '''
        Method to create a DPW, assign memory to the grids and get field values at different time and space indices
        __________________________

        Input parameters:
        --------------------------
            psi, float         : polarization angle of the incident plane wave (in radians)
            phi, float         : azimuthal angle of the incident plane wave (in radians)
            Delta_phi, float   : permissible error in the rational angle approximation to phi (in radians)
            theta, float       : polar angle of the incident plane wave (in radians)
            Delta_theta, float : permissible error in the rational angle approximation to theta (in radians)
            number, int        : number of cells in the 3D FDTD simulation
            dx, double         : separation between adjacent cells in the x direction
            dy, double         : separation between adjacent cells in the y direction
            dz, double         : separation between adjacent cells in the z direction
            dt, double         : time step for the FDTD simulation
        __________________________

        Returns:
        --------------------------
            E_fields, double array   : the electric field for the DPW as it evolves over time and space indices
            H_fields, double array   : the magnetic field for the DPW as it evolves over time and space indices
            C, double array          : stores the coefficients of the fields for the update equation of the electric fields
            D, double array          : stores the coefficients of the fields for the update equation of the magnetic fields

        '''
        self.directions, self.m = getIntegerForAngles(phi, Delta_phi, theta, Delta_theta,
                                          np.array([dx, dy, dz]))   #get the integers for the nearest rational angle
        #store max(m_x, m_y, m_z) in the last element of the array
        print(self.m)
        print(self.directions)
        self.length = int(2*np.sum(self.m[:-1])*number)                  #set an appropriate length fo the one dimensional arrays
        #the 1D grid has no ABC to terminate it, sufficiently long array prevents reflections from the back 
        #self.m = np.abs(self.m.astype(np.int32, copy=False))        #typecast to positive integers
        # Projections for field components
        projections_h, P = getProjections(psi, self.m)  #get the projection vertors for different fields
        self.projections = projections_h / np.sqrt(constants.mu_0/constants.epsilon_0) #scale the projection vector for the mangetic field
        
        if self.m[0] == 0:       #calculate dr that is needed for sourcing the 1D array
            if self.m[1] == 0:
                if self.m[2] == 0:
                    raise ValueError("not all M values can be zero")
                else:
                    self.ds = P[2]*dz/self.m[2]
            else:
                self.ds = P[1]*dy/self.m[1]
        else:
            self.ds = P[0]*dx/self.m[0]
       


class TFSFBox():
    '''
    Class to implement a Total Field/Scattered Field(TFSF) implementation of the DPW described in
    Chapter 3: Exact Total-Field/Scattered-Field Plane-Wave Source Condition
    by Tengmeng Tan and Mike Potter
    of Steven Johnson; Ardavan Oskooi; Allen Taflove, Advances in FDTD Computational Electrodynamics: Photonics and Nanotechnology,
    Artech, 2013. (ISBN:9781608071715)
    __________________________
    
    Instance variables:
    --------------------------
        n_x, int            : stores the number of grid cells along the x axis of the TFSF box
        n_y, int            : stores the number of grid cells along the y axis of the TFSF box
        n_z, int            : stores the number of grid cells along the z axis of the TFSF box
        e_x, double array   : stores the x component of the electric field for the grid cells over the TFSF box
        e_y, double array   : stores the y component of the electric field for the grid cells over the TFSF box
        e_z, double array   : stores the z component of the electric field for the grid cells over the TFSF box
        h_x, double array   : stores the x component of the magnetic field for the grid cells over the TFSF box
        h_y, double array   : stores the y component of the magnetic field for the grid cells over the TFSF box
        h_z, double array   : stores the z component of the magnetic field for the grid cells over the TFSF box
        corners, int array  : stores the coordinates of the cornets of the total field/scattered field boundaries
        time_dimension, int : stores the time length over which the FDTD simulation is run
    '''
    def __init__(self, n_x, n_y, n_z, corners, time_duration, dimensions, noOfWaves):
        '''
        Defines the instance variables of class DiscretePlaneWave()
        __________________________
        
        Input parameters:
        --------------------------
            n_x, int            : stores the number of grid cells along the x axis of the TFSF box
            n_y, int            : stores the number of grid cells along the y axis of the TFSF box
            n_z, int            : stores the number of grid cells along the z axis of the TFSF box
            corners, int array  : stores the coordinates of the cornets of the total field/scattered field boundaries
            time_dimension, int : stores the time length over which the FDTD simulation is run
        '''
        self.n_x = n_x   #assign the instance varibales with the number of grid points along each axis
        self.n_y = n_y
        self.n_z = n_z
        #intitialise the 3D grids with n_x, n_y, n_z cells and +1 components where necessary  
        self.dimensions = dimensions
        self.fields = np.zeros((noOfWaves+1, 2*dimensions, self.n_x+1, self.n_y+1, self.n_z+1), order='C')
        self.corners = corners
        self.time_duration = time_duration
        self.noOfWaves = noOfWaves
    

    def initializeABC(self):
        # Allocate memory for ABC arrays
        face_fields = np.zeros((self.noOfWaves, 4*self.dimensions, max(self.n_x, self.n_y, self.n_z),
                                max(self.n_x, self.n_y, self.n_z)), order='C')

        abccoef = (1/np.sqrt(3.0)-1.0)/(1/np.sqrt(3.0)+1.0)

        return face_fields, abccoef
        
    def getFields(self, planeWaves, snapshot, angles, number, dx, dy, dz, dt, ppw):
        face_fields, abccoef = self.initializeABC()
        for i in range(self.noOfWaves):
            planeWaves[i].runDiscretePlaneWave(angles[i, 0], angles[i, 1], angles[i, 2], angles[i, 3],
                                               angles[i, 4], number, dx, dy, dz)
            C, D = planeWaves[i].initializeGrid(np.array([dx, dy, dz]), dt)  #initialize the one dimensional arrays and coefficients
        getGridFields(planeWaves, C, D, snapshot, self.n_x, self.n_y, self.n_z, self.fields,
                      self.corners, self.time_duration, face_fields, abccoef, dt, self.noOfWaves,
                      constants.c, ppw, self.dimensions)   