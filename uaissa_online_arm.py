"""
Uncertainty Aware Implicit Safe Set Algorithm Online Stage: Manipulator Case
Nonimal policy: random exploration RL policy
"""
from envs.env_manipulator import ArmEnv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import argparse
import GPy
from IPython import embed
from IPython.display import display

initx = [np.pi/4, np.pi/4, 0, 0] # default initial state, [theta1, theta2, dtheta1, dtheta2]
np.random.seed(0) # fixed seed

# def normalize_x(theta1, theta2, dtheta1, dtheta2):

def reverse_normalize(normalize_x):
    return [normalize_x[0]*np.pi, normalize_x[1]*2*np.pi, normalize_x[2]*0.2-0.1, normalize_x[3]*0.2-0.1]

def gp_dynamics_learning(args):
    env = ArmEnv(initx)
    tau_x = args.tau_x
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

    X_tau = []
    for theta1 in theta1_set: 
        for theta2 in theta2_set: 
            for dtheta1 in dtheta1_set:
                for dtheta2 in dtheta2_set:
                    if theta1 == 0 and theta2 == 0:
                        continue
                    X_tau.append([theta1/np.pi, theta2/(2*np.pi), (dtheta1+0.1)/0.2, (dtheta2+0.1)/0.2])
    
    # for each state sample to get correct control input, select control to make delta dot d positive
    gp_dataset_control = {}
    gp_dataset_next_state = {}
    gp_dataset_deltadotd = {}
    gp_dataset_deltad = {}
    for x_tau in X_tau:
        not_found = True 
        control_deltadotd_dict = {}

        # randomly sample a control such that dot d can be increased 
        for i in range(10000):
            ddtheta1 = np.random.uniform(env.arm.ddtheta1_lim[0],env.arm.ddtheta1_lim[1])
            ddtheta2 = np.random.uniform(env.arm.ddtheta2_lim[0],env.arm.ddtheta2_lim[1])
            control = [ddtheta1, ddtheta2]
            env.reset(x_tau)
            dotd_pre = env.dotd
            d_pre = env.d
            env.step(control)
            dotd_next = env.dotd
            d_next = env.d
            delta_dotd = dotd_next - dotd_pre
            delta_d = d_next - d_pre

            # log the control
            control_deltadotd_dict[tuple([(ddtheta1+40000)/80000, (ddtheta2+40000)/80000])] = delta_dotd
            if delta_dotd > args.inf_sup_delta_dotd: 
                good_control = [(ddtheta1+40000)/80000, (ddtheta2+40000)/80000]
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
        gp_dataset_next_state[tuple(x_tau)] = [next_state[0]/np.pi, next_state[1]/(2*np.pi), (next_state[2]+0.1)/0.2, (next_state[3]+0.1)/0.2]
        gp_dataset_deltadotd[tuple(x_tau)] = delta_dotd
        gp_dataset_deltad[tuple(x_tau)] = delta_d

    # data collection finished
    print("finished colleting data")
    print("data set size is: {}".format(len(X_tau)))

    # construct GP dataset 
    train_x = np.zeros((0,args.nx + args.nu))
    train_y = np.zeros((0,args.nx))
    
    for x_tau in X_tau:
        # x_data = np.expand_dims(np.array(x_tau + gp_dataset_control[tuple(x_tau)]), dim=1)
        x_data = np.array(x_tau + gp_dataset_control[tuple(x_tau)])
        y_data = np.array(np.array(gp_dataset_next_state[tuple(x_tau)])  - np.array(x_tau))
        train_x = np.vstack((train_x, x_data))
        train_y = np.vstack((train_y, y_data))
        #import ipdb; ipdb.set_trace()


    # dynamics learning 
    gp_dynamics_dict = []
    for i in range(args.nx):
        print(i)
        print(f'gaussian process regression for the {i} dimension')
        # define the kernel 
        kernel = GPy.kern.RBF(input_dim=args.nu+args.nx, variance=0.1, lengthscale=0.1) # default kernel settings
        # gaussian process 
        # for computation efficiency, conduct separate GP for each output dimension
        import ipdb; ipdb.set_trace()
        m = GPy.models.GPRegression(train_x, np.expand_dims(train_y[:,i],1), kernel, noise_var=1e-8)
        kernel.variance.constrain_bounded(1e-8, 2.0, warning=False)
        kernel.lengthscale.constrain_bounded(1e-8, 10.0, warning=False)
        m.optimize(max_f_eval = 1000)
        display(m)
        
        gp_dynamics_dict.append(m) # append each learnt dynamics 

    

    return gp_dynamics_dict
    # with open('gp_dynamics_dict.pickle', 'wb') as handle:
    #     pickle.dump(gp_dynamics_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def safe_guard_simulation(args):
    # initialize the environment
    env = ArmEnv(initx)
    
    # load the learnt dynamics 
    # with open('gp_dynamics_dict.pickle', 'rb') as handle:
    #     gp_dynamics_dict = pickle.load(handle)

    gp_dynamics_dict = gp_dynamics_learning(args)

    # compute the well calibrated model 
    tau_munich = args.tau_munich
    M = (1 + np.floor((env.arm.theta1_lim[1] - env.arm.theta1_lim[0])/tau_munich)) \
        * (1 + np.floor((env.arm.theta2_lim[1] - env.arm.theta2_lim[0])/tau_munich)) \
        * (1 + np.floor((env.arm.dtheta1_lim[1] - env.arm.dtheta1_lim[0])/tau_munich)) \
        * (1 + np.floor((env.arm.dtheta2_lim[1] - env.arm.dtheta2_lim[0])/tau_munich)) # Munich (Uniform Error Bounds for Gaussian Process Regression with Application to Safe Control) theorem 3.1 
    delta = args.delta
    beta = 2 * np.log(M/delta)
    print("beta = {}".format(beta))


    # compute L phi 
    Ldx = max(env.arm.dtheta1_lim[1], env.arm.dtheta1_lim[1])
    Ldotdx = max(1, max(env.arm.dtheta1_lim[1], env.arm.dtheta1_lim[1]))
    Lphi = max(args.k, 1)*(Ldx + Ldotdx)

    for i in range(args.maxIter):
        ddtheta1 = np.random.uniform(env.arm.ddtheta1_lim[0],env.arm.ddtheta1_lim[1])
        ddtheta2 = np.random.uniform(env.arm.ddtheta2_lim[0],env.arm.ddtheta2_lim[1])
        u_ref = [ddtheta1,ddtheta2]
        
        # ISSA safe guard 
        # since random exploration, no need to use AdamBA to minimize the distance between
        # safe control and the reference control 
        # check the safety status on the reference control 
        x_tmp = env.state 
        w_tmp = np.array(x_tmp + u_ref)[np.newaxis, :]

        # posterior sigma
        mean_next = []
        variance = 0
        for j in range(args.nx):
            gp_dynamics = gp_dynamics_dict[j]
            next_state_j_tmp = gp_dynamics.predict(w_tmp)
            
            mean_next.append(next_state_j_tmp[0][0,0])
            variance = variance + next_state_j_tmp[1][0,0]
            
        
        # upper bound of next state phi
        ub_phi = phi(args, env, mean_next) + Lphi*beta*variance 
        
        # check safety of upper bound
        if ub_phi < max(phi(args, env, x_tmp) - args.eta, 0):
            # the reference control is safe control
            env.step(u_ref)
        else:
            # reference control is not safe 
            # generate a safe control
            safe_found = False
            for k in range(1000):
                ddtheta1 = np.random.uniform(env.arm.ddtheta1_lim[0],env.arm.ddtheta1_lim[1])
                ddtheta2 = np.random.uniform(env.arm.ddtheta2_lim[0],env.arm.ddtheta2_lim[1])
                u_new = [ddtheta1,ddtheta2]

                # compute posterior new
                mean_next_new = []
                variance_new = 0
                w_tmp = np.array(x_tmp + u_new)[np.newaxis, :]
                for j in range(args.nx):
                    gp_dynamics = gp_dynamics_dict[j]
                    next_state_j_tmp = gp_dynamics.predict(w_tmp)
                    #import ipdb; ipdb.set_trace()
                    mean_next_new.append(next_state_j_tmp[0][0,0])
                    variance_new = variance + next_state_j_tmp[1][0,0]

                ub_phi_new = phi(args, env, mean_next_new) + Lphi*beta*variance_new 
                if ub_phi_new < max(phi(args, env, x_tmp) - args.eta, 0):
                    # found a safe control point
                    u_safe = u_new
                    safe_found = True
                    break
            
            # simulate the control point
            try:
                assert(safe_found == True)
            except:
                print("zero safe control found")
                embed()
                exit()

            env.step(u_safe)
            
def safe_guard_simulation_accurate_model(args):
    env = ArmEnv(initx)

    dataset_x_delta_theta1 = np.arange(0, 1, 0.08) # dtheta1
    dataset_y_delta_theta1 = env.arm.dt * (dataset_x_delta_theta1 * 0.2 - 0.1) # delta_theta1
    kernel = GPy.kern.RBF(input_dim=1, variance=0.1, lengthscale=0.1) # default kernel settings
    m_delta_theta1 = GPy.models.GPRegression(dataset_x_delta_theta1.reshape(-1,1), dataset_y_delta_theta1.reshape(-1,1), kernel, noise_var=1e-8)
    kernel.variance.constrain_bounded(1e-2, 2.0, warning=False)
    kernel.lengthscale.constrain_bounded(1e-2, 10.0, warning=False)
    # m_delta_theta1.optimize(max_f_eval = 1000)
    display(m_delta_theta1)

    dataset_x_delta_theta2 = np.arange(0, 1, 0.08) # dtheta2
    dataset_y_delta_theta2 = env.arm.dt * (dataset_x_delta_theta2 * 0.2 - 0.1) # delta_theta2
    kernel = GPy.kern.RBF(input_dim=1, variance=0.1, lengthscale=0.1) # default kernel settings
    m_delta_theta2 = GPy.models.GPRegression(dataset_x_delta_theta2.reshape(-1,1), dataset_y_delta_theta2.reshape(-1,1), kernel, noise_var=1e-8)
    kernel.variance.constrain_bounded(1e-2, 2.0, warning=False)
    kernel.lengthscale.constrain_bounded(1e-2, 10.0, warning=False)
    # m_delta_theta2.optimize(max_f_eval = 1000)
    display(m_delta_theta2)


    dataset_x_delta_dtheta1 = np.arange(0, 1, 0.08) # ddtheta1
    dataset_y_delta_dtheta1 = env.arm.dt * (dataset_x_delta_dtheta1 * 8 - 4)  # delta_dtheta1
    kernel = GPy.kern.RBF(input_dim=1, variance=0.1, lengthscale=0.1) # default kernel settings
    m_delta_dtheta1 = GPy.models.GPRegression(dataset_x_delta_dtheta1.reshape(-1,1), dataset_y_delta_dtheta1.reshape(-1,1), kernel, noise_var=1e-8)
    kernel.variance.constrain_bounded(1e-2, 2.0, warning=False)
    kernel.lengthscale.constrain_bounded(1e-2, 10.0, warning=False)
    # m_delta_dtheta1.optimize(max_f_eval = 1000)
    display(m_delta_dtheta1)

    dataset_x_delta_dtheta2 = np.arange(0, 1, 0.08) # ddtheta2
    dataset_y_delta_dtheta2 = env.arm.dt * (dataset_x_delta_dtheta2 * 8 - 4) # delta_dtheta2
    kernel = GPy.kern.RBF(input_dim=1, variance=0.1, lengthscale=0.1) # default kernel settings
    m_delta_dtheta2 = GPy.models.GPRegression(dataset_x_delta_dtheta2.reshape(-1,1), dataset_y_delta_dtheta2.reshape(-1,1), kernel, noise_var=1e-8)
    kernel.variance.constrain_bounded(1e-2, 2.0, warning=False)
    kernel.lengthscale.constrain_bounded(1e-2, 10.0, warning=False)
    # m_delta_dtheta2.optimize(max_f_eval = 1000)
    display(m_delta_dtheta2)
    
    gp_dynamics_dict = [m_delta_theta1, m_delta_theta2, m_delta_dtheta1, m_delta_dtheta2]


    # compute the well calibrated model 
    tau_munich = args.tau_munich
    M = (1 + np.floor((0.2)/tau_munich)) \
        * (1 + np.floor((0.2)/tau_munich)) \
        * (1 + np.floor((8)/tau_munich)) \
        * (1 + np.floor((8)/tau_munich)) # Munich (Uniform Error Bounds for Gaussian Process Regression with Application to Safe Control) theorem 3.1 
    delta = args.delta
    beta = 2 * np.log(M/delta)
    print("beta = {}".format(beta))
    


    # compute L phi 
    Ldx = max(env.arm.dtheta1_lim[1], env.arm.dtheta1_lim[1])
    Ldotdx = max(1, max(env.arm.dtheta1_lim[1], env.arm.dtheta1_lim[1]))
    Lphi = max(args.k, 1)*(Ldx + Ldotdx)

    phis = []
    phis_mu = []
    phis_upperbound = []
    safe_trigger = 0
    for i in range(args.maxIter):
        print(i)
        ddtheta1 = 1e-4*np.random.uniform(env.arm.ddtheta1_lim[0],env.arm.ddtheta1_lim[1])
        ddtheta2 = 1e-4*np.random.uniform(env.arm.ddtheta2_lim[0],env.arm.ddtheta2_lim[1])
        u_ref = [ddtheta1,ddtheta2]
        
        # ISSA safe guard 
        # since random exploration, no need to use AdamBA to minimize the distance between
        # safe control and the reference control 
        # check the safety status on the reference control 
        x_tmp = env.state 
        w_tmp = np.array(x_tmp + u_ref)[np.newaxis, :]

        mu = []
        sigma = []
        for i in range(4):
            if i == 0 or i == 1:
                normalize_input = (w_tmp[0][i+2] + 0.1)*5
            elif i == 2 or i == 3:
                normalize_input = (w_tmp[0][i+2] + 4)/8
            pred_results = gp_dynamics_dict[i].predict(np.array([[normalize_input]]))
            mu.append(pred_results[0][0][0])
            sigma.append(pred_results[1][0][0])
        pred_x_next = np.array(x_tmp) + np.array(mu)
        pred_phi = phi(args, env, pred_x_next)
        ub_pred_phi = phi(args, env, pred_x_next) + Lphi*beta*np.sum(sigma)
        
        
        # upper bound of next state phi
        # next phi 
        # env.step(u_ref)
        # x_next = env.state 

        # print(pred_x_next)
        # print(x_next)
        #import ipdb; ipdb.set_trace()
        #if phi(args, env, x_next) < max(phi(args, env, x_tmp) - args.eta, 0):

        if ub_pred_phi < max(phi(args, env, x_tmp) - args.eta, 0):
            # safe control 
            phis_mu.append(pred_phi)
            phis_upperbound.append(ub_pred_phi)
            # import ipdb; ipdb.set_trace()
            env.step(u_ref)
        else:
            # unsafe control
            safe_trigger = safe_trigger + 1
            found_safe_control = False
            for k in range(1000):
                ddtheta1 = 1e-4*np.random.uniform(env.arm.ddtheta1_lim[0],env.arm.ddtheta1_lim[1])
                ddtheta2 = 1e-4*np.random.uniform(env.arm.ddtheta2_lim[0],env.arm.ddtheta2_lim[1])
                u_new = [ddtheta1,ddtheta2]

                w_tmp = np.array(x_tmp + u_new)[np.newaxis, :]
                mu = []
                sigma = []
                for i in range(4):
                    if i == 0 or i == 1:
                        normalize_input = (w_tmp[0][i+2] + 0.1)*5
                    elif i == 2 or i == 3:
                        normalize_input = (w_tmp[0][i+2] + 4)/8
                    pred_results = gp_dynamics_dict[i].predict(np.array([[normalize_input]]))
                    mu.append(pred_results[0][0][0])
                    sigma.append(pred_results[1][0][0])
                pred_x_next = np.array(x_tmp) + np.array(mu)
                pred_phi = phi(args, env, pred_x_next)
                ub_pred_phi = phi(args, env, pred_x_next) + Lphi*beta*np.sum(sigma)
                # import ipdb; ipdb.set_trace()
                if ub_pred_phi < max(phi(args, env, x_tmp) - args.eta, 0):
                    # found safe control 
                    phis_mu.append(pred_phi)
                    phis_upperbound.append(ub_pred_phi)
                    env.step(u_new)
                    found_safe_control = True
                    break
        
            if not found_safe_control:
                print('no safe control found')
                embed()
                exit()

        

        # reset if exceed
        reset_if_exceed(env)
        current_phi = phi(args, env, env.state)
        phis.append(current_phi)
    
    # plot the unicycle
    index = [i for i in range(args.maxIter)]
    print(f"env resetcound {env.resetcount}")
    print(f"safe trigger time {safe_trigger}")
    
    # plt.plot(index, np.array(phis_mu)+1)
    plt.plot(index, np.array(phis_upperbound), color='orange')
    plt.plot(index, phis, alpha=0.5, color='blue')
    plt.show()

    

def phi(args, env, x):
    d = env.the_d(x)
    dotd = env.the_dotd(x)
    dmin = 0.1
    sigma = 0 
    k = args.k
    n = 1
    phi = sigma + dmin**n - d**n - k*dotd
    return phi

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def reset_if_exceed(env):
    # reset the system if velocity exceeds limit 
    reset_flag = False
    if env.arm.dtheta1 < env.arm.dtheta1_lim[0] or env.arm.dtheta1 > env.arm.dtheta1_lim[1]:
        reset_flag = True
    if env.arm.dtheta2 < env.arm.dtheta2_lim[0] or env.arm.dtheta2 > env.arm.dtheta2_lim[1]:
        reset_flag = True
    
    if reset_flag:
        env.reset([np.pi/4, np.pi/4, 0, 0])
        env.resetcount = env.resetcount + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='The UA-ISSA Offline Stage')
    parser.add_argument('-dl','--dynamics_learning', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="compute the discretizaiton step size.")
    parser.add_argument('-sm','--simulation', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="compute the safety index design")
    parser.add_argument('-gt','--groundtruth', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="ground truth")
    parser.add_argument('--nx', type=int, nargs='?',
                         default=4, 
                         help="state dimension")
    parser.add_argument('--nu', type=int, nargs='?',
                         default=2, 
                         help="control dimension")
    parser.add_argument('--inf_sup_delta_dotd', type=float, nargs='?',
                         default=1., 
                         help="the computed infimum of supremum of Lipsthiz of delta dot d function")
    parser.add_argument('--delta', type=float, nargs='?',
                         default=0.00001, 
                         help="confidence interval")
    parser.add_argument('--eta', type=float, nargs='?',
                         default=0.05, 
                         help="state dimension")
    parser.add_argument('--tau_x', type=float, nargs='?',
                         default=0.17404655724622103, 
                         help="state discretization step size")
    parser.add_argument('--k', type=float, nargs='?',
                         default=2.540751614841154, 
                         help="safety index k design")
    parser.add_argument('--maxIter', type=int, nargs='?',
                         default=2000, 
                         help="maximum simulation iteration")
    parser.add_argument('--tau_munich', type=float, nargs='?',
                         default=0.000001, 
                         help="the tau value of Munich paper")


    args = parser.parse_args()

    # if args.dynamics_learning:
    #     gp_dynamics_learning(args)

    # if args.simulation:
    #     safe_guard_simulation(args)

    if args.groundtruth:
        safe_guard_simulation_accurate_model(args)

