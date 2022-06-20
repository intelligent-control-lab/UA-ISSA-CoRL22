"""
Uncertainty Aware Implicit Safe Set Algorithm Offline Stage: Manipulator Case
"""
from envs.env_manipulator import ArmEnv
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import GPy
from IPython import embed

initx = [np.pi/4, np.pi/4, 0, 0] # default initial state, [theta1, theta2, dtheta1, dtheta2]
np.random.seed(0) # fixed seed

def uaissa_offline_collect_data(args):
    ''' Collect Gaussian process data on the discretized state
    '''
    # compute the state discretization step size 
    # exponentially lower the discretization step size 
    tau_x = 0.5 # discretization step size 
    env = ArmEnv(initx)
    # disretize state space [px, py, theta, v, w]
    theta1_lim = env.arm.theta1_lim
    theta2_lim = env.arm.theta2_lim
    dtheta1_lim = env.arm.dtheta1_lim
    dtheta2_lim = env.arm.dtheta2_lim

    while True:
    #     # collect GP samples
    #     # disretize state space according to the current tau_x
        theta1_set = np.arange(theta1_lim[0], theta1_lim[1], tau_x)
        theta2_set = np.arange(theta2_lim[0], theta2_lim[1], tau_x)
        dtheta1_set = np.arange(dtheta1_lim[0], dtheta1_lim[1], tau_x)
        dtheta2_set = np.arange(dtheta2_lim[0], dtheta2_lim[1], tau_x)
        size = theta1_set.shape[0]*theta2_set.shape[0]*dtheta1_set.shape[0]*dtheta2_set.shape[0]

        X_tau = []
        for theta1 in theta1_set: 
            for theta2 in theta2_set: 
                for dtheta1 in dtheta1_set:
                    for dtheta2 in dtheta2_set:
                        if theta1 == 0 and theta2 == 0:
                            continue
                        X_tau.append([theta1,theta2,dtheta1,dtheta2])
        
        # for each state sample to get correct control input, select control to make delta dot d positive
        gp_dataset_control = {}
        gp_dataset_next_state = {}
        gp_dataset_deltadotd = {}
        for x_tau in X_tau:
            not_found = True 
            control_deltadotd_dict = {}

            # randomly sample a control such that dot d can be increased 
            #print('----')
            for i in range(10000):
                ddtheta1 = np.random.uniform(env.arm.ddtheta1_lim[0],env.arm.ddtheta1_lim[1])
                ddtheta2 = np.random.uniform(env.arm.ddtheta2_lim[0],env.arm.ddtheta2_lim[1])
                control = [ddtheta1,ddtheta2]
                env.reset(x_tau)
                dotd_pre = env.dotd
                env.step(control)
                dotd_next = env.dotd
                delta_dotd = dotd_next - dotd_pre


                # log the control 
                control_deltadotd_dict[tuple([ddtheta1, ddtheta2])] = delta_dotd
                # if delta_dotd > 0:
                #print(delta_dotd)
                if delta_dotd > args.inf_sup_delta_dotd: 
                    good_control =control
                    next_state = env.state
                    not_found = False
                    break
            
            try:
                assert not_found == False # definitely should find good control 
            except:
                print("safe control not found")
                embed()
                exit()

            gp_dataset_control[tuple(x_tau)] = good_control
            gp_dataset_next_state[tuple(x_tau)] = next_state
            gp_dataset_deltadotd[tuple(x_tau)] = delta_dotd
        
        print("finished colleting data")
        print("data set size is: {}".format(len(X_tau)))
        # construct GP dataset 
        train_x = np.zeros((0,args.nx + args.nu))
        train_y = np.zeros((0,args.nx))

        for x_tau in X_tau:
            # x_data = np.expand_dims(np.array(x_tau + gp_dataset_control[tuple(x_tau)]), dim=1)
            x_data = np.array(x_tau + gp_dataset_control[tuple(x_tau)])
            y_data = np.array(gp_dataset_next_state[tuple(x_tau)])
            train_x = np.vstack((train_x, x_data))
            train_y = np.vstack((train_y, y_data))

    #     # conduct gaussian process 
    #     # we only need to posterior K, which only relates to inputs. Hence we simply predict one dimension
        kernel = GPy.kern.RBF(input_dim=args.nu+args.nx, variance=0.0001, lengthscale=0.1) # default kernel settings
    #     # kernel.variance.constrain_bounded(1e-8, 0.1, warning=False)
    #     # kernel.lengthscale.constrain_bounded(1e-8, 1, warning=False)
    #     # m = GPy.models.GPRegression(train_x, np.expand_dims(train_y[:,0],1), kernel)
    #     # m.optimize(max_f_eval = 1000) # posterior optimization 


        # determine if tau is small enough
        delta_d_list = []
        for key in gp_dataset_deltadotd.keys():
            delta_d_list.append(gp_dataset_deltadotd[key])
        # inf_sup_delta_dotd = min(delta_d_list)
        # print("inf_sup_delta_dotd = {}".format(inf_sup_delta_dotd))
        # inf_sup_delta_dotd = 1
        #sup_delta_dotd = max(delta_d_list)
        Ldx = max(env.arm.dtheta1_lim[1], env.arm.dtheta1_lim[1])
        Ldotdx = max(1, max(env.arm.dtheta1_lim[1], env.arm.dtheta1_lim[1]))
        Lf = args.nx + args.nu*env.arm.dt
        nx = args.nx
        tau_munich = 0.01
        M = (1 + np.floor((env.arm.theta1_lim[1] - env.arm.theta1_lim[0])/tau_munich)) \
            * (1 + np.floor((env.arm.theta2_lim[1] - env.arm.theta2_lim[0])/tau_munich)) \
            * (1 + np.floor((env.arm.dtheta1_lim[1] - env.arm.dtheta1_lim[0])/tau_munich)) \
            * (1 + np.floor((env.arm.dtheta2_lim[1] - env.arm.dtheta2_lim[0])/tau_munich)) # Munich (Uniform Error Bounds for Gaussian Process Regression with Application to Safe Control) theorem 3.1 
        delta = args.delta
        beta = 2 * np.log(M/delta)
        print("beta = {}".format(beta))
        # beta = ((env.arm.theta1_lim[1] - env.arm.theta1_lim[0])/tau_munich) * (env.arm.theta0_lim[1]/tau_munich)
        l = float(kernel.lengthscale)
        sigma = np.sqrt(float(kernel.variance))

        # first edition
        # Z_star = np.min((l**2, env.arm.ddtheta1_lim[1]**2, env.arm.ddtheta2_lim[1]**2))
        # L_k = (sigma ** 2 / l ** 2) * np.sqrt(np.exp(-Z_star/l**2) * Z_star) # Lk for the kernel matrix
        # K = kernel.K(x_data)
        # N_x = X_tau.shape[0]
        # tilde_sigma = 2*L_k*tau_x + 2*N_x*L_k*tau_x*np.linalg.norm(np.linalg.inv(K))*sigma**2 # 1st posterior sigma

        # second edition
        tilde_sigma = sigma**2 - sigma**2 * np.exp(-(tau_x**2)/(2*l**2))**2 # 2nd posterior sigma 
        
        import ipdb; ipdb.set_trace()
        tau_condition = args.inf_sup_delta_dotd - (Ldx + Ldotdx)*tau_x - (Ldx \
                        + Ldotdx)*Lf*tau_x - 2*(Ldx + Ldotdx)*beta*nx*tilde_sigma # general form of eq. 67 (the coefficient for k should be positive)

        print("---------- tau condition value: {} ---------".format(tau_condition))
        if tau_condition > 0:
            print("discretization step size is chosen successfully as {}".format(tau_x))
            print("Resulted GP datasize = {}".format(size))
            exit(0)
            break 
        else:
            # not small enough tau, keep diminish discretization step size 
            tau_x = tau_x * 0.99
            print("discretizaiton is too large, try {}".format(tau_x))

    return tau_x

def safety_index_design(args, tau_x):
    env = ArmEnv(initx)
    # disretize state space [px, py, theta, v, w]
    theta1_lim = env.arm.theta1_lim
    theta2_lim = env.arm.theta2_lim
    dtheta1_lim = env.arm.dtheta1_lim
    dtheta2_lim = env.arm.dtheta2_lim

    # collect discretization state
    theta1_set = np.arange(theta1_lim[0], theta1_lim[1], tau_x)
    theta2_set = np.arange(theta2_lim[0], theta2_lim[1], tau_x)
    dtheta1_set = np.arange(dtheta1_lim[0], dtheta1_lim[1], tau_x)
    dtheta2_set = np.arange(dtheta2_lim[0], dtheta2_lim[1], tau_x)
    size = theta1_set.shape[0]*theta2_set.shape[0]*dtheta1_set.shape[0]*dtheta2_set.shape[0]

    X_tau = []
    for theta1 in theta1_set: 
        for theta2 in theta2_set: 
            for dtheta1 in dtheta1_set:
                for dtheta2 in dtheta2_set:
                    if theta1 == 0 and theta2 == 0:
                        continue
                    X_tau.append([theta1,theta2,dtheta1,dtheta2])
    
    # for each state sample to get correct control input, select control to make delta dot d positive
    gp_dataset_control = {}
    gp_dataset_next_state = {}
    gp_dataset_deltadotd = {}
    gp_dataset_deltad = {}
    for x_tau in X_tau:
        not_found = True 
        control_deltadotd_dict = {}

        # randomly sample a control such that dot d can be increased 
        #print('----')
        for i in range(10000):
            ddtheta1 = np.random.uniform(env.arm.ddtheta1_lim[0],env.arm.ddtheta1_lim[1])
            ddtheta2 = np.random.uniform(env.arm.ddtheta2_lim[0],env.arm.ddtheta2_lim[1])
            control = [ddtheta1,ddtheta2]
            env.reset(x_tau)
            dotd_pre = env.dotd
            d_pre = env.d
            env.step(control)
            dotd_next = env.dotd
            d_next = env.d
            delta_dotd = dotd_next - dotd_pre
            delta_d = d_next - d_pre


            # log the control 
            control_deltadotd_dict[tuple([ddtheta1, ddtheta2])] = delta_dotd
            if delta_dotd > args.inf_sup_delta_dotd: 
                good_control =control
                next_state = env.state
                not_found = False
                break
        
        try:
            assert not_found == False # definitely should find good control 
        except:
            print("safe control not found")
            embed()
            exit()

        gp_dataset_control[tuple(x_tau)] = good_control
        gp_dataset_next_state[tuple(x_tau)] = next_state
        gp_dataset_deltadotd[tuple(x_tau)] = delta_dotd
        gp_dataset_deltad[tuple(x_tau)] = delta_d

    # data collection finished
    # define the kernel 
    kernel = GPy.kern.RBF(input_dim=args.nu+args.nx, variance=0.0001, lengthscale=0.1) # default kernel settings

    # evaluate on each discretized state 
    Ldx = max(env.arm.dtheta1_lim[1], env.arm.dtheta1_lim[1])
    Ldotdx = max(1, max(env.arm.dtheta1_lim[1], env.arm.dtheta1_lim[1]))
    Lf = args.nx + args.nu*env.arm.dt
    nx = args.nx
    tau_munich = 0.01
    M = (1 + np.floor((env.arm.theta1_lim[1] - env.arm.theta1_lim[0])/tau_munich)) \
        * (1 + np.floor((env.arm.theta2_lim[1] - env.arm.theta2_lim[0])/tau_munich)) \
        * (1 + np.floor((env.arm.dtheta1_lim[1] - env.arm.dtheta1_lim[0])/tau_munich)) \
        * (1 + np.floor((env.arm.dtheta2_lim[1] - env.arm.dtheta2_lim[0])/tau_munich)) # Munich (Uniform Error Bounds for Gaussian Process Regression with Application to Safe Control) theorem 3.1 
    delta = args.delta
    beta = 2 * np.log(M/delta)
    print("beta = {}".format(beta))
    l = float(kernel.lengthscale)
    sigma = np.sqrt(float(kernel.variance))

    # second edition
    tilde_sigma = sigma**2 - sigma**2 * np.exp(-(tau_x**2)/(2*l**2))**2 # 2nd posterior sigma 

    # compute safety index k on each discretized state 
    k_candidates = []
    for x_tau in X_tau:
        numerator = args.eta - gp_dataset_deltad[tuple(x_tau)]
        denominator = gp_dataset_deltadotd[tuple(x_tau)] - (Ldx + \
                        Ldotdx)*tau_x - (Ldx + Ldotdx)*Lf*tau_x - \
                        2*(Ldx + Ldotdx)*beta*nx*tilde_sigma 
        k_tmp = max(1, numerator/denominator)
        k_candidates.append(k_tmp)

    # select the maximum k 
    k = max(k_candidates)
    print(f"the safety index design is: {k}")
    return k

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
    parser.add_argument('-cds','--compute_tau_x', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="compute the discretizaiton step size.")
    parser.add_argument('-csi','--compute_safety_index', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="compute the safety index design")
    parser.add_argument('--inf_sup_delta_dotd', type=float, nargs='?',
                         default=1., 
                         help="the computed infimum of supremum of Lipsthiz of delta dot d function")
    parser.add_argument('--nx', type=int, nargs='?',
                         default=4, 
                         help="state dimension")
    parser.add_argument('--nu', type=int, nargs='?',
                         default=2, 
                         help="control dimension")
    parser.add_argument('--delta', type=float, nargs='?',
                         default=0.05, 
                         help="confidence interval")
    parser.add_argument('--eta', type=float, nargs='?',
                         default=0.05, 
                         help="state dimension")

    args = parser.parse_args()

    if args.compute_tau_x:
        tau_x = uaissa_offline_collect_data(args)

    tau_x = 0.17404655724622103 # precomputed discretization step size 
    if args.compute_safety_index:
        k = safety_index_design(args, tau_x)



    

