import numpy as np
from read_data import *
from scipy.interpolate import BSpline
from scipy.optimize import minimize

def loss_function_RNA_expression(w, x, y, knots, degree, K, lmbda):
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
    mse = np.linalg.norm(error, ord=2)
    reg = lmbda * w.dot(K).dot(w)
    return mse + reg 


def x_fit_bspline(T, rna_expression, knots, lmbda=1e4, k=3):
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
    output = []

    # fitting 
    for i in range(len(rna_expression[0])):
        w = fit_bspline(T, rna_expression[:,i], knots, k, lmbda)
        output.append(BSpline(knots, w, k))

    return output

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

    # Create K matrix (filled with integrated B-spline basis functions)
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
    result = minimize(loss_function_RNA_expression, w0, args=(x, y, knots, degree, K, lmbda))

    # Get the optimal weights and calculate the predicted values
    w = result.x

    return w

def loss_function_protein_expression(alpha, r, lmbda_prot, mRNA_expression, knots, K, T, degree, gamma_alpha):
    bspl = BSpline(knots, alpha/r, degree)
    bspl_der = bspl.derivative(nu=1) # first derivative
    loss = np.linalg.norm((-mRNA_expression + lmbda_prot*bspl(T) + bspl_der(T)), ord=2) + gamma_alpha*alpha.dot(K).dot(alpha)
    # loss = np.linalg.norm(r*mRNA_expression - lmbda_prot*bspl(T) - bspl_der(T), ord=2) - alpha.dot(K).dot(alpha)
    return loss


def fit_bspline_protein_expression(r, lmbda_prot, T, knots, degree, number_of_basis_functions, gene_expression):

    # Set the B-spline parameters
    K = np.zeros((number_of_basis_functions, number_of_basis_functions))

    # Create K matrix (filled with integrated B-spline basis functions)
    for j in range(number_of_basis_functions):
        for k in range(number_of_basis_functions):
            w_j = [0 if i != j else 1 for i in range(number_of_basis_functions)]
            w_k = [0 if i != k else 1 for i in range(number_of_basis_functions)]
            bspl_second_order_der_j = BSpline(knots, w_j, degree).derivative(nu=2)
            bspl_second_order_der_k = BSpline(knots, w_k, degree).derivative(nu=2)
            phi_j = bspl_second_order_der_j(T)
            phi_k = bspl_second_order_der_k(T)
            K[j, k] = np.sum(phi_j*phi_k)

    # Initialize the weights
    alpha = np.zeros(number_of_basis_functions)

    # regularization to make function smooth (and > 0)
    gamma_alpha = 100

    # Minimize the loss function
    # result = minimize(loss_function_protein_expression, alpha, args=(A, K, gamma))
    result = minimize(loss_function_protein_expression, alpha, args=(r, lmbda_prot, gene_expression, knots, K, T, 3, gamma_alpha))
    # Get the optimal weights and calculate the predicted values
    w = result.x

    return w

def loss_function_protein_expression_microRNA(w, r, lmbda_prot, mRNA_expression, knots, K, T, degree, gamma_alpha, microRNA_expression, gamma_z):
    z_i = w[:len(microRNA_expression)]
    alpha = w[len(microRNA_expression):]
    bspl = BSpline(knots, alpha, degree)
    bspl_der = bspl.derivative(nu=1) # first derivative
    loss = np.linalg.norm((-r*mRNA_expression + mRNA_expression*microRNA_expression.dot(z_i) + lmbda_prot*bspl(T) + bspl_der(T)), ord=2) + gamma_alpha*alpha.dot(K).dot(alpha)

def fit_bspline_protein_expression_microRNA(r, lmbda_prot, T, knots, degree, number_of_basis_functions, mRNA_expression, microRNA_expression):
    """COMMENT: make microRNA expression the expression of microRNA genes for all time points
    ADD INEQUALITY CONSTRAINTS

    Args:
        r (_type_): _description_
        lmbda_prot (_type_): _description_
        T (_type_): _description_
        knots (_type_): _description_
        degree (_type_): _description_
        number_of_basis_functions (_type_): _description_
        mRNA_expression (_type_): _description_
        microRNA_expression (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Set the B-spline parameters
    K = np.zeros((number_of_basis_functions, number_of_basis_functions))

    # Create K matrix (filled with integrated B-spline basis functions)
    for j in range(number_of_basis_functions):
        for k in range(number_of_basis_functions):
            w_j = [0 if i != j else 1 for i in range(number_of_basis_functions)]
            w_k = [0 if i != k else 1 for i in range(number_of_basis_functions)]
            bspl_second_order_der_j = BSpline(knots, w_j, degree).derivative(nu=2)
            bspl_second_order_der_k = BSpline(knots, w_k, degree).derivative(nu=2)
            phi_j = bspl_second_order_der_j(T)
            phi_k = bspl_second_order_der_k(T)
            K[j, k] = np.sum(phi_j*phi_k)

    # Initialize the weights
    alpha = np.zeros(number_of_basis_functions + len(microRNA_expression))

    # regularization to make function smooth (and > 0)
    gamma_alpha = 100

    # Minimize the loss function
    result = minimize(loss_function_protein_expression_microRNA, alpha, args=(r, lmbda_prot, mRNA_expression, knots, K, T, 3, gamma_alpha))
    # Get the optimal weights and calculate the predicted values
    w = result.x

    return w




