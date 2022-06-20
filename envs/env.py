from .unicycle import Unicycle
import numpy as np

class Env():
    def __init__(self, x):
        self.obsp = [1,1] # obstacle position 
        self.obsr = 0.1 # obstacle radius
        self.car = Unicycle(x) 

    def step(self, u):
        '''
        One step forward 
        Input: 
            u: control [F_L - F_R]
        '''
        self.car.step(u)

    def reset(self, x):
        '''
        Reset states
        Input:
            x: state (px, py, theta, v, w)
        '''
        self.car.reset(x)

    @property
    def d(self):
        distance = np.linalg.norm(np.array(self.obsp) - np.array([self.car.px, self.car.py]))
        return distance

    @property
    def dotd(self):
        vec_theta = np.array([np.cos(self.car.theta), np.sin(self.car.theta)])
        vec_car2obs = np.array([self.obsp[0] - self.car.px, self.obsp[1] - self.car.py])
        cosalpha = np.dot(vec_theta,vec_car2obs) / (np.linalg.norm(vec_theta) * np.linalg.norm(vec_car2obs))
        dotdistance = -self.car.v * cosalpha
        return dotdistance

    @property
    def state(self):
        x = [self.car.px, self.car.py, self.car.theta, self.car.v, self.car.w]
        return x
