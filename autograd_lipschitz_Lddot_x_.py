import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever need
from tqdm import tqdm
# constant
ox, oy = 1, 1
obsr = 0.1
M = 1 # mass
J = 0.1 # inertia moment
l = 1 # half length of axis 
Bv = 1 # translation friction 
Bw = 0.1 # rotation friction 
dt = 0.001 # simulation time step size

# state limit variables
px_lim = [0,2] # px limitation [px_min, px_max]
py_lim = [0,2] # py limitation [py_min, py_max]
theta_lim = [0,2*np.pi] # theta
v_lim = [0,1] # v limitation [v_min, v_max]
w_lim = [-0.4,0.4] # w limitation [w_min, w_max]

# control limit variables
Fl_lim = [-1,1]
Fr_lim = [-1,1]


def dotd(x):
    # state
    px = x[0]
    py = x[1]
    theta = x[2]
    v = x[3]
    w = x[4]

    vec_theta = np.array([np.cos(theta), np.sin(theta)])
    vec_car2obs = np.array([ox - px, oy - py])
    cosalpha = np.dot(vec_theta,vec_car2obs) / (np.linalg.norm(vec_theta) * np.linalg.norm(vec_car2obs))
    dotd = v * cosalpha
    return dotd

def main():
    tau_x = 0.2
    tau_u = 0.2
    px_set = np.arange(px_lim[0], px_lim[1], 0.5)
    py_set = np.arange(py_lim[0], py_lim[1], 0.5)
    theta_set = np.arange(theta_lim[0],theta_lim[1], np.pi/4)
    v_set = np.arange(v_lim[0], v_lim[1], 0.01)
    w_set = np.arange(w_lim[0], w_lim[1], 0.1)

    X_tau = []
    for px in px_set: 
        for py in py_set: 
            for theta in theta_set:
                for v in v_set:
                    for w in w_set:
                        X_tau.append(np.array([px,py,theta,v,w]))

    
    grad_dotd_func = grad(dotd)
    Lipcshitz_dotd = 0
    for i in tqdm(range(len(X_tau))):
        input_for_dotd = X_tau[i]
        grad_dotd = grad_dotd_func(input_for_dotd)
        grad_list = np.array([grad_dotd])
        L = np.sum(np.abs(grad_list))
        if L > Lipcshitz_dotd:
            Lipcshitz_dotd = L
            print(input_for_dotd)
            print(grad_list)
            print(Lipcshitz_dotd)
        if i % 100 == 0:
            print(Lipcshitz_dotd)
        #print(grad_px)
        

    #import ipdb; ipdb.set_trace()


main()

