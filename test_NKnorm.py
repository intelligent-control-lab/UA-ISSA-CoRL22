import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import GPy
from IPython.display import display
from tqdm import tqdm



class Toy_mobile_robot():
    def __init__(self, x_pos=0, y_pos=0, vel=0, vel_angle=0, dt=0.1):
        self.init_x_pos = x_pos
        self.init_y_pos = y_pos
        self.init_vel = vel
        self.init_vel_angle = vel_angle
        self.dt = dt
        self.x = None
        self.reset()

    def reset(self):
        self.x = [self.init_x_pos, self.init_y_pos, self.init_vel, self.init_vel_angle]  # vel_angle in radians

        return self.x

    def step(self, u):
        """
        :param x: [x,y,v,vel_angle]
        :param u: [a,w]
        :param dt: 0.01
        :return: x_{t+1}
        x_{t+1} =
            [ x + v * cos(vel_angle) ,
              y + v * sin(vel_angle) ,
              v + a * dt ,
              alpha + w * dt ]
        """
        x_pos = self.x[0]
        y_pos = self.x[1]
        vel = self.x[2]
        vel_angle = self.x[3]

        acc = np.min((u[0], 10))
        acc_angle = np.min((u[1], 10))

        next_state = [x_pos + vel * self.dt * np.cos(vel_angle),
                      y_pos + vel * self.dt * np.sin(vel_angle),
                      np.min((vel + np.sin(acc) * self.dt, 10)),
                      np.clip(vel_angle + acc_angle * self.dt, -np.pi/2, np.pi/2)
                      ]

        self.x = next_state

        return self.x


def main():
    # # Constant
    # dt = 0.1

    # # np.random.seed(1)
    # env = Toy_mobile_robot(dt=dt)
    # MAX_episode = 100
    # MAX_timestep = 100

    # DATA_SET = []


    # for ep in range(MAX_episode):
    #     print('Episode {}'.format(ep))
    #     x = env.reset()
    #     for t in range(MAX_timestep):
    #         u = [np.random.uniform(-10, 10), np.random.uniform(-10, 10)]
    #         new_x = env.step(u)

    #         # store to data set
    #         data_sample = [ np.array([x[2], x[3], u[0], u[1]]), np.array(new_x) - np.array(x)] # (1000, 2, 4)
    #         DATA_SET.append(data_sample)
    #         # print(data_sample)
    #         x = new_x

    r = 1
    tau = 0.1
    N_x = (r/tau)
    DATA_SET_X = np.mgrid[0:r:tau]
    DATA_SET_X = DATA_SET_X.reshape(-1,1)
    kernel = GPy.kern.RBF(input_dim=1, variance=0.01, lengthscale=0.1)
    # m = GPy.models.GPRegression(DATA_SET_X[0:data_size], kernel)
    K_inv = np.linalg.inv(kernel.K(DATA_SET_X))
    K_inv_norm = np.linalg.norm(K_inv)
    print("Size {}:   || K^-1 ||: {}    N*|| K^-1 || = {}   tau*N*|| K^-1 || = {}".format(N_x, K_inv_norm, N_x * K_inv_norm, tau*N_x * K_inv_norm))

    r = 1
    tau = 0.01
    N_x = (r/tau)
    DATA_SET_X = np.mgrid[0:r:tau]
    DATA_SET_X = DATA_SET_X.reshape(-1,1)
    kernel = GPy.kern.RBF(input_dim=1, variance=0.01, lengthscale=0.1)
    # m = GPy.models.GPRegression(DATA_SET_X[0:data_size], kernel)
    K_inv = np.linalg.inv(kernel.K(DATA_SET_X))
    K_inv_norm = np.linalg.norm(K_inv)
    print("Size {}:   || K^-1 ||: {}    N*|| K^-1 || = {}   tau*N*|| K^-1 || = {}".format(N_x, K_inv_norm, N_x * K_inv_norm, tau*N_x * K_inv_norm))

        
    #     data_sample = [np.array([]) , ]
    
    # DATA_SET = np.array(DATA_SET)
    # DATA_SET_X = DATA_SET[:, 0:1] 
    # DATA_SET_Y = DATA_SET[:, 1:] 
    # DATA_SET_X = DATA_SET_X.reshape(-1, DATA_SET_X.shape[-1])
    # DATA_SET_Y = DATA_SET_Y.reshape(-1, DATA_SET_Y.shape[-1])

    

    #import ipdb; ipdb.set_trace()
    
    # 100 data size

    # # 300 data size
    # data_size = 300
    # kernel = GPy.kern.RBF(input_dim=4, variance=0.01, lengthscale=0.1)
    # # m = GPy.models.GPRegression(DATA_SET_X[0:data_size], kernel)
    # K_inv = np.linalg.pinv(kernel.K(DATA_SET_X[0:data_size]))
    # K_inv_norm = np.linalg.norm(K_inv)
    # print("Size {}:   || K^-1 ||: {}    N*|| K^-1 || = {}".format(data_size, K_inv_norm, data_size * K_inv_norm))

    # # 1000 data size
    # data_size = 1000
    # kernel = GPy.kern.RBF(input_dim=4, variance=0.01, lengthscale=0.1)
    # # m = GPy.models.GPRegression(DATA_SET_X[0:data_size], kernel)
    # K_inv = np.linalg.pinv(kernel.K(DATA_SET_X[0:data_size]))
    # K_inv_norm = np.linalg.norm(K_inv)
    # print("Size {}:   || K^-1 ||: {}    N*|| K^-1 || = {}".format(data_size, K_inv_norm, data_size * K_inv_norm))

    # # 3000 data size
    # data_size = 3000
    # kernel = GPy.kern.RBF(input_dim=4, variance=0.01, lengthscale=0.1)
    # # m = GPy.models.GPRegression(DATA_SET_X[0:data_size], kernel)
    # K_inv = np.linalg.pinv(kernel.K(DATA_SET_X[0:data_size]))
    # K_inv_norm = np.linalg.norm(K_inv)
    # print("Size {}:   || K^-1 ||: {}    N*|| K^-1 || = {}".format(data_size, K_inv_norm, data_size * K_inv_norm))


main()