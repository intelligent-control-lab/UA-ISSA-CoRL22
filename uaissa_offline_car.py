"""
Uncertainty Aware Implicit Safe Set Algorithm Offline Stage
"""
from envs.env import Env
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import GPy
from IPython import embed

initx = [0,0,0,0,0] # default initial state 

def uaissa_offline_collect_data(args):
    ''' Collect Gaussian process data on the discretized state
    '''
    # compute the state discretization step size 
    # exponentially lower the discretization step size 
    tau_x = 0.5
    env = Env(initx)
    # disretize state space [px, py, theta, v, w]
    px_lim = env.car.px_lim
    py_lim = env.car.py_lim
    theta_lim = np.array([0., 2*np.pi])
    v_lim = env.car.v_lim
    w_lim = env.car.w_lim

    while True:
        # collect GP samples
        # disretize state space according to the current tau_x
        px_set = np.arange(px_lim[0], px_lim[1], tau_x)
        py_set = np.arange(py_lim[0], py_lim[1], tau_x)
        theta_set = np.arange(theta_lim[0],theta_lim[1], tau_x)
        v_set = np.arange(v_lim[0], v_lim[1], tau_x)
        w_set = np.arange(w_lim[0], w_lim[1], tau_x)
        # w_set = [0]

        X_tau = []
        for px in px_set: 
            for py in py_set: 
                for theta in theta_set:
                    for v in v_set:
                        for w in w_set:
                            if px == 1 and py == 1:
                                continue
                            X_tau.append(np.array([px,py,theta,v,w]))
        
        # for each state sample to get correct control input, select control to make delta dot d positive
        gp_dataset_control = {}
        gp_dataset_next_state = {}
        gp_dataset_deltadotd = {}
        for x_tau in X_tau:
            not_found = True 
            control_deltadotd_dict = {}

            # randomly sample a control
            for i in range(1000):
                Fl = np.random.uniform(env.car.Fl_lim[0],env.car.Fl_lim[1])
                Fr = np.random.uniform(env.car.Fr_lim[0],env.car.Fr_lim[1])

                control = [Fl, Fr]
                env.reset(x_tau)
                dotd_pre = env.dotd
                env.step(control)
                dotd_next = env.dotd
                delta_dotd = dotd_next - dotd_pre


                # log the control 
                control_deltadotd_dict[tuple([Fl, Fr])] = delta_dotd
                if delta_dotd > 0:
                    good_control =control
                    next_state = env.state
                    not_found = False
                    break
            
            try:
                assert not_found == False # definitely should find good control 
            except:
                print("control not found")
                embed()
                exit()

            gp_dataset_control[tuple(x_tau)] = good_control
            gp_dataset_next_state[tuple(x_tau)] = next_state
            gp_dataset_deltadotd[tuple(x_tau)] = delta_dotd
        
         
        # construct GP dataset 
        train_x = np.zeros((0,x_tau.len()+good_control.len()))
        train_y = np.zeros((0,x_tau.len()))

        for x_tau in X_tau:
            x_data = np.array(x_tau + gp_dataset_control[tuple(x_tau)])
            y_data = np.array(gp_dataset_next_state[tuple(x_tau)])
            train_x = np.vstack(train_x, x_data)
            train_y = np.vstack(train_y, y_data)

        # conduct gaussian process 
        # we only need to posterior K, hence simply predict one dimension
        kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
        kernel.variance.constrain_bounded(1e-8, 0.1, warning=False)
        kernel.lengthscale.constrain_bounded(1e-8, 1, warning=False)
        m = GPy.models.GPRegression(x_data, np.expand_dims(y_data[:,0],1), kernel)
        m.optimize(max_f_eval = 1000) # posterior optimization 


        # determine if tau is small enough
        delta_d_list = []
        for key in gp_dataset_deltadotd.keys():
            delta_d_list.append(gp_dataset_deltadotd[key])
        inf_sup_delta_dotd = min(delta_d_list)
        sup_delta_dotd = max(delta_d_list)
        Ldx = 1
        Ldotdx = 1
        Lf = 5
        nx = args.nx
        beta = args.beta
        l = float(kernel.lengthscale)
        sigma = np.sqrt(float(kernel.variance))
        Z_star = np.min((l**2, env.car.Fl_lim[1]**2, env.car.Fr_lim[1]**2))
        L_k = (sigma ** 2 / l ** 2) * np.sqrt(np.exp(-Z_star/l**2) * Z_star) # Lk for the kernel matrix
        K = kernel.K(x_data)
        N_x = X_tau.len()
        tau_condition = inf_sup_delta_dotd / (2*(Ldx+Ldotdx)*(1+Lf+2*beta*nx*np.sqrt(2*L_k*tau_x + 2*N_x*L_k*tau_x*np.linalg.norm(np.linalg.inv(K))*sigma**2)))

        # Gaussian process on the dataset 
        print("data collected successfully!")
        exit()
 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The UA-ISSA Offline Stage')
    parser.add_argument('-cod','--collect_data', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate collect data for dynamics learning.")
    parser.add_argument('--nu', type=int, nargs='?', 
                        default=2, 
                        help="the dimension of the control")
    parser.add_argument('--inf_sup_delta_dotd', type=float, nargs='?',
                         default=1., 
                         help="the computed infimum of supremum of Lipsthiz of delta dot d function")
    parser.add_argument('--nx', type=int, nargs='?',
                         default=5, 
                         help="state dimension")
    parser.add_argument('--beta', type=int, nargs='?',
                         default=1, 
                         help="state dimension")

    args = parser.parse_args()

    if args.collect_data:
        uaissa_offline_collect_data(args)

    

