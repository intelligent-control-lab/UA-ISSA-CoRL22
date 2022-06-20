import numpy as np

class Arm():
    def __init__(self, x):
        '''
        Initialization
        Input:
            x: state (theta1, theta2, dtheta1, dtheta2)
        '''
        self.l1 = 1 # link 1 length
        self.l2 = 1 # link 2 length
        self.dt = 0.001 # simulation time step size

        # state limit variables
        self.theta1_lim = [0.1, np.pi-0.1] # theta1 limitation [theta1_min, theta1_max]
        self.theta2_lim = [0, 2*np.pi] # theta2 limitation [theta2_min, theta2_max]
        self.dtheta1_lim = [-0.1, 0.1] # dtheta1 limitation [dtheta1_min, dtheta1_max]
        self.dtheta2_lim = [-0.1, 0.1] # dtheta2 limitation [dtheta2_min, dtheta2_max]

        # control limit variables
        self.ddtheta1_lim = [-40000,40000]
        self.ddtheta2_lim = [-40000,40000]
        # self.ddtheta1_lim = [-4,4]
        # self.ddtheta2_lim = [-4,4]

        # initilization 
        self.theta1 = x[0]
        self.theta2 = x[1]
        self.dtheta1 = x[2]
        self.dtheta2 = x[3]

    def reset(self, x):
        '''
        Reset states
        Input:
            x: state (px, py, theta, v, w)
        '''
        self.theta1 = x[0]
        self.theta2 = x[1]
        self.dtheta1 = x[2]
        self.dtheta2 = x[3]

    def step(self, u):
        '''
        One step forward 
        Input: 
            u: control [ddtheta1, ddtheta2]
        '''
        ddtheta1 = u[0]
        ddtheta2 = u[1]

        # update state variables
        self.theta1 = self.theta1 + self.dtheta1*self.dt
        self.theta2 = self.theta2 + self.dtheta2*self.dt
        self.dtheta1 = self.dtheta1 + ddtheta1*self.dt
        self.dtheta2 = self.dtheta2 + ddtheta2*self.dt

        # clip state
        self.theta1 = np.mod(self.theta1, np.pi) 
        self.theta1 = np.clip(self.theta1, self.theta1_lim[0], self.theta1_lim[1])
        self.theta2 = np.mod(self.theta2, 2*np.pi)
        # self.dtheta1 = np.clip(self.dtheta1, self.dtheta1_lim[0], self.dtheta1_lim[1])
        # self.dtheta2 = np.clip(self.dtheta2, self.dtheta2_lim[0], self.dtheta2_lim[1])







    

