from .manipulator import Arm
import numpy as np

class ArmEnv():
    def __init__(self, x):
        self.wall = 1.6 # position on the 1.6m wrt. origin
        self.arm = Arm(x) 
        self.resetcount = 0

    def step(self, u):
        '''
        One step forward 
        Input: 
            u: control [F_L - F_R]
        '''
        self.arm.step(u)

    def reset(self, x):
        '''
        Reset states
        Input:
            x: state (px, py, theta, v, w)
        '''
        self.arm.reset(x)

    @property
    def d(self):
        distance = self.wall - self.arm.l1*np.cos(self.arm.theta1) - self.arm.l2*np.cos(self.arm.theta2)
        return distance

    @property
    def dotd(self):
        dot_distance = self.arm.l1*np.sin(self.arm.theta1)*self.arm.dtheta1 + self.arm.l2*np.sin(self.arm.theta2)*self.arm.dtheta2
        return dot_distance

    @property
    def state(self):
        x = [self.arm.theta1, self.arm.theta2, self.arm.dtheta1, self.arm.dtheta2]
        return x


    def the_d(self, x):
        theta1 = x[0]
        theta2 = x[1]
        distance = self.wall - self.arm.l1*np.cos(theta1) - self.arm.l2*np.cos(theta2)
        return distance 

    def the_dotd(self, x):
        theta1 = x[0]
        theta2 = x[1]
        dtheta1 = x[2]
        dtheta2 = x[3]
        dot_distance = self.arm.l1*np.sin(theta1)*dtheta1 + self.arm.l2*np.sin(theta2)*dtheta2
        return dot_distance 

