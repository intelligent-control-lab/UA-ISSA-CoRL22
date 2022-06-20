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
    # Constant
    dt = 0.1

    # np.random.seed(1)
    env = Toy_mobile_robot(dt=dt)
    MAX_episode = 10
    MAX_timestep = 100

    DATA_SET = []


    for ep in range(MAX_episode):
        print('Episode {}'.format(ep))
        x = env.reset()
        for t in range(MAX_timestep):
            u = [np.random.uniform(-10, 10), np.random.uniform(-10, 10)]
            new_x = env.step(u)

            # store to data set
            data_sample = [ np.array([x[2], x[3], u[0], u[1]]), np.array(new_x) - np.array(x)] # (1000, 2, 4)
            DATA_SET.append(data_sample)
            # print(data_sample)
            x = new_x
    DATA_SET = np.array(DATA_SET)
    DATA_SET_X = DATA_SET[:, 0:1, 0:2] # v, theta
    DATA_SET_Y = DATA_SET[:, 1:, 0:1] # delta px
    DATA_SET_X = DATA_SET_X.reshape(-1, DATA_SET_X.shape[-1])
    DATA_SET_Y = DATA_SET_Y.reshape(-1, DATA_SET_Y.shape[-1])
    print(DATA_SET.shape)
    print(DATA_SET[3])
    print(DATA_SET_X.shape)
    print(DATA_SET_X[3])
    print(DATA_SET_Y.shape)
    print(DATA_SET_Y[3])
    # import ipdb; ipdb.set_trace()

    # X = np.random.uniform(-3., 3., (20, 1))
    # print(X.shape)
    # # GP
    kernel = GPy.kern.RBF(input_dim=2, variance=0.01, lengthscale=0.1)
    kernel.variance.constrain_bounded(1e-8, 1.0, warning=False)
    kernel.lengthscale.constrain_bounded(1e-8, 10.0, warning=False)
    
    #kernel = GPy.kern.Matern52(2,ARD=True) + GPy.kern.White(2)
    # define kernel
    # ker = GPy.kern.Matern52(2, ARD=True) + GPy.kern.White(2)

    m = GPy.models.GPRegression(DATA_SET_X, DATA_SET_Y,kernel)
    # m.kern.variance.set_prior(GPy.priors.Gamma(0.1, 0.1))



    ##### visuliazation ########
    display(m)
    # ax = m.plot()
    # dataplot = ax['dataplot'][0]
    # dataplot.figure.savefig("test_figure/1d_vpredict.pdf")
    # optimized_results = m.optimize(max_f_eval = 1000)
    
    m.optimize(max_f_eval = 1000)
    # m.optimize_restarts(num_restarts=10)
    display(m)
    ax = m.plot()
    # print(ax)
    dataplot = ax['dataplot'][0]
    dataplot.figure.savefig("test_figure/2d_px_predict_verify_optimized.pdf")
    GP_prediction = m.predict(np.array([[0.5, 0.5]]))


    ####### Efficiently Verify the Uniform Error Bound #######
    # parameters and constant 
    

    ########## TODO  #########
    # get a_max, sigma, l , dt, N, delta
    v_max = 10.0
    acc_max = 10.0
    acc_angle_max = 10.0
    theta_max = np.pi/2
    sigma = np.sqrt(float(kernel.variance))
    l = float(kernel.lengthscale)
    dt = dt
    N = MAX_episode * MAX_timestep
    delta = 0.01 # 1 - delta = 95%

    ########## TODO  #########
    L_f = dt * v_max if v_max > 1 else dt # function Lipchitz, here we have f(v, theta) = dt * v * cos(theta), the corresponding Lipchitz constant in (v, theta) \in (-1, 1) (-\pi, \pi) is 
    
    # B_star = np.max((-a_max, -l)) # the minimizer of derivative of kernel
    # RBF (aka. squared exponential kernel)
    # L_k = (-2 * sigma ** 2 * np.exp(-B_star**2/(2 * l**2)) * B_star) / (l**2) # kernel Lipchitz 
    Z_star = np.min((l**2, v_max**2, theta_max**2))
    # L_k = sigma ** 2 / (l * np.exp(1/2))
    L_k = (sigma ** 2 / l ** 2) * np.sqrt(np.exp(-Z_star/l**2) * Z_star)
    tau = 1e-60 # grid constant
    M_tau = (np.floor(2*theta_max / tau) + 1.) * (np.floor(2*v_max / tau) + 1.) # minimum grid sampling number
    print("M_tau = {}".format(M_tau))
    
    k_max = sigma**2 * np.exp(0) # maximum kernel 

    # Compute constant 
    L_vn = L_k * np.sqrt(N) * np.linalg.norm(np.linalg.pinv(kernel.K(DATA_SET_X)) * DATA_SET_Y)
    w_tau = np.sqrt(2*tau*L_k*(1 + N * np.linalg.norm(np.linalg.pinv(kernel.K(DATA_SET_X))) * k_max))
    beta_tau = 2*np.log(M_tau / delta)
    gamma_tau = (L_vn + L_f) * tau + np.sqrt(beta_tau) * w_tau
    # import ipdb; ipdb.set_trace()

    # uniform error bound 
    # TODO: error_bound = np.sqrt(beta_tau) * MODEL.Delta(x) + gamma_tau 

    ########## Test error bound  #########
    cnt_in_error_bound = 0 
    cnt_out_error_bound = 0

    v_check_points = np.arange(np.min(DATA_SET_X, axis=0)[0], np.max(DATA_SET_X, axis=0)[0], 0.1)
    theta_check_points = np.arange(np.min(DATA_SET_X, axis=0)[1], np.max(DATA_SET_X, axis=0)[1], 0.01)
    max_error_ratio = -1
    max_error_bound = -1
    max_error_bound_error_abs = -1

    error_ground_truth = []
    error_bound_ground_truth = []
    for v in tqdm(v_check_points):
        for theta in theta_check_points:

            pred_results = m.predict(np.array([[v, theta]]))
            pred_mean = pred_results[0][0][0]
            pred_variance = pred_results[1][0][0]
            pred_sigma = np.sqrt(pred_variance)

            ground_truth = v*np.cos(theta)*dt

            

            error_abs = abs(ground_truth - pred_mean)
            error_bound = np.sqrt(beta_tau) * pred_sigma + gamma_tau 

            error_ground_truth.append(error_abs/abs(ground_truth))
            error_bound_ground_truth.append(error_bound/abs(ground_truth))

            print("------")
            print((v, theta))
            print(pred_results)
            print(error_abs)
            print(error_bound)
            if error_abs/error_bound > max_error_ratio:
                max_error_ratio = error_abs/error_bound
            print(error_abs/error_bound)
            print(max_error_ratio)

            if error_abs <= error_bound:
                cnt_in_error_bound += 1
            else:
                cnt_out_error_bound += 1

            if error_bound > max_error_bound:
                max_error_bound = error_bound
                max_error_bound_error_abs = error_abs


    print("In error bound {} Times".format(cnt_in_error_bound))
    print("Out error bound {} Times".format(cnt_out_error_bound))
    print("In error bound percentage = {}%".format((cnt_in_error_bound/(cnt_in_error_bound+cnt_out_error_bound))*100))

    print("max (error/error_bound) = {}".format(max_error_ratio))
    print("max error bound = {}".format(max_error_bound))
    print("max error bound error abs = {}".format(max_error_bound_error_abs))

    print("max error/ground_truth = {}".format(np.max(error_ground_truth)))
    print("max error_bound/ground_truth ={}".format(np.max(error_bound_ground_truth)))
    print("min error/ground_truth = {}".format(np.min(error_ground_truth)))
    print("min error_bound/ground_truth ={}".format(np.min(error_bound_ground_truth)))
    # import ipdb; ipdb.set_trace()
        # pred_mean = 





if __name__ == '__main__':
    main()
