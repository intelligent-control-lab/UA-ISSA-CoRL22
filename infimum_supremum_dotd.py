"""
The implementation of Theorem 3 on dynamics model of unicycle
"""
# from envs.unicycle import Unicycle
from xmlrpc.client import boolean
from envs.env import Env
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import argparse
import GPy

initx = [0,0,0,0,0] # default initial state 

def collect_delta_d_dataset(args):
    env = Env(initx)
    # disretize state space [px, py, theta, v, w]
    px_lim = env.car.px_lim
    py_lim = env.car.py_lim
    theta_lim = np.array([0., 2*np.pi])
    v_lim = env.car.v_lim
    w_lim = env.car.w_lim

    # state and control discretization step size 
    tau_x = args.tau_x
    tau_u = args.tau_u
    
    # disretize state space
    px_set = np.arange(px_lim[0], px_lim[1], tau_x)
    py_set = np.arange(py_lim[0], py_lim[1], tau_x)
    theta_set = np.arange(theta_lim[0],theta_lim[1], tau_x)
    v_set = np.arange(v_lim[0], v_lim[1], tau_x)
    w_set = np.arange(w_lim[0], w_lim[1], tau_x)

    X_tau = []
    for px in px_set: 
        for py in py_set: 
            for theta in theta_set:
                for v in v_set:
                    for w in w_set:
                        X_tau.append(np.array([px,py,theta,v,w]))

    # discretize control space 
    u1_set = np.arange(env.car.Fl_lim[0], env.car.Fl_lim[1], tau_u)
    u2_set = np.arange(env.car.Fr_lim[0], env.car.Fr_lim[1], tau_u)

    # discretized control space
    U_tau = []
    for u1 in u1_set: 
        for u2 in u2_set:
            U_tau.append(np.array([u1,u2]))
    
    # collect dataset for each discretized state 
    trainX_dictionary = {} # the key is discretized state 
    trainY_dictionary = {} # the key is discretized state 

    for x_tau in tqdm(X_tau):
        # collect data for this state 
        trainX = np.zeros([0, 2]) # 2 dimensional control 
        trainY = np.zeros([0, 1]) # 1 dimensional output
        for control in U_tau:
            # reset state 
            env.reset(x_tau)
            dotd_pre = env.dotd
            #print([env.car.px, env.car.py, env.car.theta, env.car.v, env.car.w])
            # simulate step
            env.step(control)
            #print([env.car.px, env.car.py, env.car.theta, env.car.v, env.car.w])
            dotd_next = env.dotd
            delta_dotd = dotd_next - dotd_pre
            # expand trainX and trainY 
            trainX = np.vstack((trainX, control))
            trainY = np.vstack((trainY, delta_dotd))

        
        # expand the dicitonary
        trainX_dictionary[tuple(x_tau)] = trainX
        trainY_dictionary[tuple(x_tau)] = trainY

    # save the trainX and trainY dictionary 
    with open('trainX_dict.pickle', 'wb') as handle:
        pickle.dump(trainX_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('trainY_dict.pickle', 'wb') as handle:
        pickle.dump(trainY_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('trainX_dict_toy.pickle', 'wb') as handle:
    #     pickle.dump(trainX_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # with open('trainY_dict_toy.pickle', 'wb') as handle:
    #     pickle.dump(trainY_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

def gaussian_process_discretized_state(args):
    env = Env(initx)

    # state and control discretization step size 
    tau_x = args.tau_x
    tau_u = args.tau_u

    # load the saved gp dataset 
    with open('trainX_dict_toy.pickle', 'rb') as handle:
        trainX = pickle.load(handle)
    with open('trainY_dict_toy.pickle', 'rb') as handle:
        trainY = pickle.load(handle)

    kernels_dict = {}
    supdotd_dict = {}
    # gp on each discretized state 
    for key in trainX.keys():
        x_data = trainX[key]
        y_data = trainY[key]
        kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
        kernel.variance.constrain_bounded(1e-8, 0.1, warning=False)
        kernel.lengthscale.constrain_bounded(1e-8, 1, warning=False)
        m = GPy.models.GPRegression(x_data, y_data, kernel)
        m.optimize(max_f_eval = 1000) # posterior optimization 
        kernels_dict[key] = m

        # compute the delta d max for each discretized state
        # constant computation
        l = float(kernel.lengthscale)
        sigma = np.sqrt(float(kernel.variance))
        Z_star = np.min((l**2, env.car.Fl_lim[1]**2, env.car.Fr_lim[1]**2))
        L_k = (sigma ** 2 / l ** 2) * np.sqrt(np.exp(-Z_star/l**2) * Z_star) # Lk for the kernel matrix
        Nhat = y_data.shape[0]
        K = kernel.K(x_data)
        r = np.abs(np.max((env.car.Fl_lim[1]-env.car.Fl_lim[0], env.car.Fr_lim[1]-env.car.Fr_lim[0]))) # maximum extension of the control space 

        # intermediate term 
        bar_sigma = np.sqrt((max(y_data) + np.sqrt(Nhat)*L_k*tau_u*np.linalg.norm(np.linalg.inv(K)*y_data))**2 + 2*L_k*tau_u + 2*Nhat*L_k*tau_u*np.linalg.norm(np.linalg.inv(K))*sigma**2)
        
        # sup dot d 
        if bar_sigma > np.sqrt(r*L_k):
            constant = 1
        else:
            constant = 0
        supdotd = 12*np.sqrt(args.nu*r*L_k) * (np.sqrt(2*np.log(np.exp*2**1.5)) + constant * np.log((bar_sigma**2)/ (r*L_k))) - bar_sigma*np.sqrt(2*np.log(1/args.sigma_L))

        # save the dictionary of supdotd wrt. discretized state 
        supdotd_dict[key] = supdotd

    # save the sup dot d dictionary 
    with open('supdotd_dict.pickle', 'wb') as handle:
        pickle.dump(supdotd_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # compute the infimum of supremum of dot d 
    supre_dotd_collection = []
    for key in supdotd_dict.keys():
        supre_dotd_collection.append(supdotd_dict[key])
    inf_sup_dotd = min(supre_dotd_collection) - args.L_delta_dotd

    print(f'the infimum of supremum dot d is: {inf_sup_dotd}')
        

def simulate_env():
    env = Env(initx)
    """
    # plot the unicycle
    for i in range(100):
        env.step(u)
        # print(f'the x pos:{env.car.px}, the y pos:{env.car.py}, heading angle:{env.car.theta}, velocity: {env.car.v}')
        print(f'the distance: {env.d}, the dot distance: {env.dotd}' )
        # print(car.py)
        plt.axis([0, 0.01, 0, 0.01])
        plt.scatter(env.car.px, env.car.py)
        plt.pause(0.05)
        plt.clf()
    """
 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The Gaussian process procedure step')
    parser.add_argument('-cod','--collect_data', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate collect data for delta dot d.")
    parser.add_argument('-gcd','--gp_collected_data', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate Gaussian process for each discretized state.")
    parser.add_argument('-cdd','--compute_delta_dotd', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate computation for delta dotd")
    parser.add_argument('--tau_u', type=float, nargs='?', default=0.2)
    parser.add_argument('--tau_x', type=float, nargs='?', default=0.2)
    parser.add_argument('--sigma_L', type=float, nargs='?', default=0.05)
    parser.add_argument('--nu', type=int, nargs='?', default=2, help="the dimension of the control")
    parser.add_argument('--L_delta_dotd', type=float, nargs='?', default=1., help="Lipsthiz of delta dot d function")

    args = parser.parse_args()

    # specify the value 
    args.tau_x = 0.2
    args.tau_u = 0.2
    # args.tau_x = 1.
    # args.tau_u = 0.5
    args.nu = 2

    if args.collect_data:
        collect_delta_d_dataset(args)
    
    if args.gp_collected_data:
        gaussian_process_discretized_state(args)
