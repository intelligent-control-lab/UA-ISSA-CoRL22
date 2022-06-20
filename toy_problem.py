import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import GPy
from IPython.display import display

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

        acc = u[0]
        acc_angle = u[1]

        next_state = [x_pos + vel * self.dt * np.cos(vel_angle),
                      y_pos + vel * self.dt * np.sin(vel_angle),
                      vel + acc * self.dt,
                      vel_angle + acc_angle * self.dt
                      ]

        self.x = next_state

        return self.x


def main():
    env = Toy_mobile_robot()
    MAX_episode = 10
    MAX_timestep = 100

    DATA_SET = []


    for ep in range(MAX_episode):
        print('Episode {}'.format(ep))
        x = env.reset()
        for t in range(MAX_timestep):
            u = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
            new_x = env.step(u)

            # store to data set
            data_sample = [ np.array([x[2], x[3], u[0], u[1]]), np.array(new_x) - np.array(x)]
            DATA_SET.append(data_sample)
            # print(data_sample)
            x = new_x
    DATA_SET = np.array(DATA_SET)
    DATA_SET_X = DATA_SET[:, 0:1, 0:2]
    DATA_SET_Y = DATA_SET[:, 1:, 0:1]
    DATA_SET_X = DATA_SET_X.reshape(-1, DATA_SET_X.shape[-1])
    DATA_SET_Y = DATA_SET_Y.reshape(-1, DATA_SET_Y.shape[-1])
    print(DATA_SET.shape)
    print(DATA_SET[3])
    print(DATA_SET_X.shape)
    print(DATA_SET_X[3])
    print(DATA_SET_Y.shape)
    print(DATA_SET_Y[3])

    # X = np.random.uniform(-3., 3., (20, 1))
    # print(X.shape)
    # # GP
    kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
    #kernel = GPy.kern.Matern52(2,ARD=True) + GPy.kern.White(2)
    # define kernel
    # ker = GPy.kern.Matern52(2, ARD=True) + GPy.kern.White(2)

    m = GPy.models.GPRegression(DATA_SET_X, DATA_SET_Y,kernel)

    display(m)
    ax = m.plot()
    dataplot = ax['dataplot'][0]
    dataplot.figure.savefig("gp-test_dim2.pdf")

    m.optimize(max_f_eval = 1000)
    display(m)
    ax = m.plot()
    print(ax)
    dataplot = ax['dataplot'][0]
    dataplot.figure.savefig("gp-test_optimized_dim2.pdf")

    GP_prediction = m.predict(np.array([[0, 1]]))
    print(GP_prediction)
    # GPy.plotting.show(fig, filename='basic_gp_regression_density_notebook_optimized')



if __name__ == '__main__':
    main()
