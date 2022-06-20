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


def sample_inf_sup(args):
    tau_x = 0.5
    env = ArmEnv(initx)
    theta1_lim = env.arm.theta1_lim
    theta2_lim = env.arm.theta2_lim
    dtheta1_lim = env.arm.dtheta1_lim
    dtheta2_lim = env.arm.dtheta2_lim

    for i in range(4):
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
        sup_delta_dotd_set = []
        for x_tau in X_tau:

            # randomly sample a control such that dot d can be increased 
            #print('----')
            sup_delta_dotd = -np.inf
            for i in range(1000):
                ddtheta1 = np.random.uniform(env.arm.ddtheta1_lim[0],env.arm.ddtheta1_lim[1])
                ddtheta2 = np.random.uniform(env.arm.ddtheta2_lim[0],env.arm.ddtheta2_lim[1])
                control = [ddtheta1,ddtheta2]
                env.reset(x_tau)
                dotd_pre = env.dotd
                env.step(control)
                dotd_next = env.dotd
                delta_dotd = dotd_next - dotd_pre

                if delta_dotd > sup_delta_dotd:
                    sup_delta_dotd = delta_dotd
            
            try:
                assert sup_delta_dotd > 0. # definitely should find good control 
            except:
                print("no positive delta dot d found")
                embed()
                exit()
            sup_delta_dotd_set.append(sup_delta_dotd)
        
        # get the infimum of all the sup delta d
        inf_sup_delta_dotd = min(sup_delta_dotd_set)
        print(f'discretization step size is {tau_x}')
        print(f'the inf_sup is {inf_sup_delta_dotd}')
        tau_x = tau_x / 2



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

    sample_inf_sup(args)


    

