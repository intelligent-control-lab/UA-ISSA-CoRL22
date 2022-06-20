import numpy as np

class Unicycle():
    def __init__(self, x):
        '''
        Initialization
        Input:
            x: state (px, py, theta, v, w)
        '''
        self.M = 1 # mass
        self.J = 0.1 # inertia moment
        self.l = 1 # half length of axis 
        self.Bv = 1 # translation friction 
        self.Bw = 0.1 # rotation friction 
        self.dt = 0.001 # simulation time step size

        # state limit variables
        self.px_lim = [0,2] # px limitation [px_min, px_max]
        self.py_lim = [0,2] # py limitation [py_min, py_max]
        self.theta_lim = [0,2*np.pi] # theta
        self.v_lim = [-1,1] # v limitation [v_min, v_max]
        self.w_lim = [-0.4,0.4] # w limitation [w_min, w_max]

        # control limit variables
        self.Fl_lim = [-100,100]
        self.Fr_lim = [-100,100]

        # initilization  
        self.px = x[0]
        self.py = x[1]
        self.theta = x[2]
        self.v= x[3]
        self.w = x[4] 

    def reset(self, x):
        '''
        Reset states
        Input:
            x: state (px, py, theta, v, w)
        '''
        self.px = x[0]
        self.py = x[1]
        self.theta = x[2]
        self.v= x[3]
        self.w = x[4] 

    def step(self, u):
        '''
        One step forward 
        Input: 
            u: control [F_L - F_R]
        '''
        Fl = u[0]
        Fr = u[1]

        # acceleration and dot w
        F = Fl + Fr
        T = self.l * (Fr - Fl)
        acc = (F - self.Bv*self.v) / self.M
        dotw = (T - self.Bw*self.w) / self.J

        # update state variables
        self.px = self.px + self.v*np.cos(self.theta)*self.dt
        self.py = self.py + self.v*np.sin(self.theta)*self.dt
        self.theta = self.theta + self.w*self.dt
        self.v = self.v + acc*self.dt
        self.w = self.w + dotw*self.dt

        # # clip state
        self.px = np.clip(self.px, self.px_lim[0], self.px_lim[1])
        self.py = np.clip(self.py, self.py_lim[0], self.py_lim[1])
        self.theta = np.mod(self.theta, 2*np.pi) 
        # self.v = np.clip(self.v, self.v_lim[0], self.v_lim[1])
        self.w = np.clip(self.w, self.w_lim[0], self.w_lim[1])







    

