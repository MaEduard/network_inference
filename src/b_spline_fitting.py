import numpy as np
from read_data import *
from scipy.interpolate import BSpline
from scipy.optimize import minimize

### Customized fitting ###

# Define the loss function
def loss_function(w, x, y, knots, degree, K, lmbda):
    """Loss function P1

    Args:
        w (float[]): weights to be optimized
        x (float): time points data
        y (float[]): ground truth gene expression
        knots (float[]): set of knot points
        degree (int): degree of B-spline basis functions
        K (float[][]): 2D roughtness matrix
        lmbda (float): regularization parameter (gamma_theta in paper)

    Returns:
        float: loss 
    """
    bspl = BSpline(knots, w, degree) # outputs function f(x) = sum(c * B_i_p(x))
    y_pred = bspl(x)    
    error = y - y_pred
    mse = np.sqrt(np.mean(error**2))
    reg = lmbda * w.dot(K).dot(w)
    return mse + reg    

def fit_bspline(x, y, knots, degree, lmbda):
    """fits B spline basis function by minimizing regularized MSE

    Args:
        x (float[]): time points
        y (float[]): normalized gene expression data
        knots (int): number of knots used
        degree (int): degree of B-spline basis functions
        lmbda (float): regularization parameter (gamma_theta in paper)

    Returns:
        float[]: weights used for B-spline weighting in linear sum of B-spline basis functions
    """
    # Set the B-spline parameters
    number_of_basis_functions = len(knots) - degree - 1
    K = np.zeros((number_of_basis_functions, number_of_basis_functions))

    # Create K matrix (filled with integrated )
    for j in range(number_of_basis_functions):
        for k in range(number_of_basis_functions):
            w_j = [0 if i != j else 1 for i in range(number_of_basis_functions)]
            w_k = [0 if i != k else 1 for i in range(number_of_basis_functions)]
            bspl_second_order_der_j = BSpline(knots, w_j, degree).derivative(nu=2)
            bspl_second_order_der_k = BSpline(knots, w_k, degree).derivative(nu=2)
            phi_j = bspl_second_order_der_j(x)
            phi_k = bspl_second_order_der_k(x)
            K[j, k] = np.sum(phi_j*phi_k)

    # Initialize the weights
    w0 = np.zeros(len(knots) - degree - 1)

    # Minimize the loss function
    result = minimize(loss_function, w0, args=(x, y, knots, degree, K, lmbda))

    # Get the optimal weights and calculate the predicted values
    w = result.x

    return w

def x_fit_bspline(networks, lmbda=1e4, number_of_knots=9, k=3):
    """Fits x data with linear sum of B-spline basis functions as: x(t) = SUM(w*B_spl(k, knots, networks, lmbda)) in which 
    knots are uniformly distributed, networks are the control points, k is the degree of the B-spline, and lmbda is the regularization constant.

    Args:
        networks (float[][]): 2D matrix of time point (row = time point) and normalized network gene expression (column = network)  (as described at https://www.synapse.org/#!Synapse:syn3049712/wiki/74633)
        lmbda (float, optional): L2 regularization factor (gamma_theta in paper). Default is 1e4.
        number_of_knots (int, optional): number of knots needed to fit the B-spline basis functions. k + D + 1 = number_of_knots. So D (number of basis functions used) is indirectly chosen by this parameter. Defaults to 9.
        k (int, optional): degree of B-spline basis function. Default is 3.

    Returns:
        BSpline[]: list of BSpline functions per network with size (1, networks#columns)
    """
    # data and hyperparameters
    x = networks[:,0]
    output = []

    # fitting 
    for i in range(len(networks[0])):
        knots = np.linspace(min(x), max(x), number_of_knots)
        w = fit_bspline(x, networks[:,i], knots, k, lmbda)
        output.append(BSpline(knots, w, k))

    return output

def get_bspline_vector(networks, number_of_knots, k):
    """Generates 2 vectors that contain the basis b-spline function and the derivative of the b-spline basis function, respectively. They each have length
    D (number of basis functions) and every entry in a vector thus contains the i'th basis function of weighted sum that the SciPy BSpline normally returns. 
    Individual Bspline basis functions are non-zero on certain domains described by the knots of the Bspline (https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-basis.html)
    More specifically, the i'th Bspline basis function is non zero on the interval [t_j, t_{j+k+1}) where t_j describes the j'th knot and k describes the degree of
    the basis function. Of note, we do not actually use the gene expression data itself to construct these B-spline functions, rather the time points given in the `networks` matrix.

    Args:
        networks (float[]): gene expression data that represent the anchor points 
        number_of_knots (int): number of knots to be used --> uniform knot distribution is used 
        k (int): degree of B-spline basis functions

    Returns:
        ([Bspline()], [Bspline().derivative()]): B-spline basis functions, and derivative functions
    """

    bspl_vec = []
    bspl_der_vec = []

    x = networks[:,0]
    knots = np.linspace(min(x), max(x), number_of_knots)
    D = number_of_knots - k - 1
    bspl = BSpline(t=knots, k=k, c=[1 for i in range(D)])
    bspl_derivative = bspl.derivative() # first order derivative
    non_zero_intervals_basis_functions = [[knots[i] for i in range(j, j+k+1+1)] for j in range(D)] # non zero interval for basis function j is [t_j, t_{j+k+1}) but since range function excludes the last value you add +1

    for i in range(len(non_zero_intervals_basis_functions)):
        bspl_vec.append(bspl.basis_element(non_zero_intervals_basis_functions[i]))
        bspl_der_vec.append(bspl_derivative.basis_element(non_zero_intervals_basis_functions[i]))
    
    return bspl_vec, bspl_der_vec

def b_vec(t, bspl_vec, bspl_derivative_vec, lmbda_PROT):
    phi_t = np.array([bspl_vec[i](t) for i in range(len(bspl_vec))])
    phi_der_t = np.array([bspl_derivative_vec[i](t) for i in range(len(bspl_derivative_vec))])
    b = lmbda_PROT*phi_t + phi_der_t

    return b

### SCIPY fitting ###

# from scipy.interpolate import splrep
# Plot scipy curve fitting
# x = n1[:,0]
# x_new = np.linspace(min(x), max(x), 100)
# nets = [1, 2, 3, 4, 5]
# colors = ['b', 'r', 'lime', 'magenta', 'black']
# markers = ["o", "d", "v", "s", "*", "^", "o", "d", "v", "s", "*", "^"]
# for i in range(len(nets)):
#     tck = splrep(x, n1[:,nets[i]], k=3, s=1)
#     # tck = splrep(x, n1[:,nets[i]], k=3, task=-1, t=x[1:len(x)-1:3]) # note that the number of knots must be equal or smaller to to k = D - 1 + P + 1  and this function does not allow the outside data points to be taken into account. This minimizes the square loss.  
   
#     y_fit = BSpline(*tck)(x_new)
#     plt.plot(x, n1[:,nets[i]], label=f"gene {nets[i]}", color=colors[i], linewidth=1.0)
#     plt.plot(x_new, y_fit, '--', color=colors[i], linewidth=1.0)

# plt.title("Network 1 of insilico_size10_2 for s = 1 and k = 3")
# plt.ylabel("Expresssion levels")
# plt.xlabel("Time")
# plt.ylim([0, 1.1])
# plt.legend(loc='upper right', fancybox = True, shadow=True)
# # plt.grid()
# plt.show()

