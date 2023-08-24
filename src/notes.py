import cvxpy as cp
import numpy as np

# # Problem data.
# m = 15
# n = 10
# np.random.seed(1)
# A = np.zeros((m, n))
# A[:, :(n//2)] = np.random.rand(n//2)
# A[:, (n//2):] = np.random.rand(n//2)*-1

# # Construct the problem.
# x = cp.Variable(n)
# objective = cp.Minimize(cp.sum_squares(A @ x) + cp.norm1(x[(n//2):]))  # Adding L1 norm regularization
# constraints = [0 <= x[:(n//2)], x[:(n//2)] <= x[(n//2):], x[n//2] == 1]
# prob = cp.Problem(objective, constraints)

# # The optimal objective value is returned by `prob.solve()`.
# result = prob.solve()
# # The optimal value for x is stored in `x.value`.
# print(x.value)

# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
# print(constraints[0].dual_value)

import numpy as np
from scipy.optimize import minimize

# Define your objective function
def objective(x, p):
    a = x[:n]
    b = x[n:]
    # p = np.array([[-0.84612997, -1.57316407, -0.01657866,  1.50485703,  3.01603091,
    #      0.10968808, -0.78214964, -0.25183436, -0.24545759,  0.40345275],
    #    [-0.11514957,  1.09823878, -1.32732243, -1.74937287,  1.6746182 ,
    #      0.17173892, -0.31965546, -0.14078842,  0.45149977, -0.70535948],
    #    [-2.17552064,  0.21857097,  0.26564888, -0.03602693, -0.35710684,
    #      1.80982095, -0.03717935,  0.56271399, -1.22534203,  0.61960289],
    #    [-0.76108116,  0.48183867, -0.57016123,  0.41577579, -0.28134718,
    #     -0.66893386,  1.17967049, -0.12811354, -0.966751  ,  0.48082406],
    #    [ 1.14597636,  0.62705032, -0.14259081,  0.30175692,  1.24092173,
    #     -0.60107449, -0.90591693,  0.86901064,  0.57759233,  2.90516521],
    #    [ 0.99135928, -0.50464841, -0.64110906,  1.22513993, -0.28818956,
    #      0.21314823, -1.70067442, -1.18533149, -2.00566251, -0.84404852],
    #    [ 1.58645431,  0.31061673,  0.01294548,  0.40454908, -1.65750261,
    #      0.40932775,  0.74734169, -1.5463409 , -2.01379216, -1.27731646],
    #    [-1.29942729,  1.24207483,  0.09162428,  0.91248716,  0.45561096,
    #      0.46991343, -1.24506827,  0.90953708,  1.3911809 , -1.14851371],
    #    [-1.4580837 ,  1.53256386,  1.08359306,  2.32037725, -0.74902525,
    #     -1.27450538, -0.41404022, -1.72733538, -1.91350897,  0.27986947],
    #    [-0.11876918,  0.05380048, -0.47930159, -2.51297341,  0.47796167,
    #      0.40830974,  1.73908258,  0.73423562,  0.58849675,  0.25071025]])
    u = 0.16502191
    return np.sum(p.dot(a) - p.dot(b)*u)**2 - np.linalg.norm(b, ord=2)**2

# Define your constraint functions
def constraint1(x):
    return x[n] - 1  # Constraint: The first entry in b should be 1

def constraint2(x):
    a = x[:n]
    b = x[n:]
    return b - a # Constraint: 0 <= a <= b

# Define the number of variables
n = 100  # Adjust this to match the dimension of your vectors

# Initial guess for optimization variables (a and b concatenated)
x0 = np.ones(2 * n)

# Define the bounds for variables a and b
bounds = [(0, None)] * n + [(1, None)] + [(0, None)] * (n-1)

# Define the constraints
constraints = [
    {'type': 'eq', 'fun': constraint1},
    {'type': 'ineq', 'fun': constraint2}
]

# Solve the optimization problem
p = np.random.randn(100,100)
result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, args=(p))

# Print the optimization result
print(result)


# from scipy.optimize import minimize
# def c1(X):
#     'Constraint on total cost to plant.'
#     x, y = X
#     return -(120 * x + 210 * y - 15000)

# def c2(X):
#     'Storage constraint'
#     x, y = X
#     return -(110 * x + 30 * y - 4000)

# def c3(X):
#     'Land area constraint'
#     x, y = X
#     return -(x + y - 75)

# def c4(X):
#     'positivity constraint'
#     return X[0]

# def c5(X):
#     'positivity constraint'
#     return X[1]

# def profit(X):
#     'Profit function'
#     x, y = X
#     return -(143 * x + 60 * y)

# sol = minimize(profit, [60, 15], constraints=[{'type': 'ineq', 'fun': f} for f in [c1, c2, c3, c4, c5]])
# print(sol)

# from itertools import chain, combinations

# def powerSet(iterable):
#    """Computes the powerset of a given iterable of numbers. Filters out any set bigger than size 3 as the paper restricts
#    genes to be regulated only by at max 2 proteins to reduce the complexity of the model.

#    Example: [1, 2, 3] --> [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3)]

#    Args:
#        iterable (int[]): list of genes

#    Returns:
#        tuple[]: list of possible gene combinations that can regulate a gene
#    """
#    s = list(iterable)
#    powerset = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
#    shrinked_powerset = [item for item in powerset if len(item) < 3]
#    return shrinked_powerset

# if __name__ == "__main__":
#     a = {1, 2, 3, 4}
#     b = powerSet(a)
#     print(b)

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import BSpline
# from scipy.interpolate import splrep, splev

# # Generate example data points
# x_data = np.array([0, 1, 2, 3, 4, 5, 6])
# y_data = np.array([1, 2, 3, 2, 1, 3, 2])

# # Define the number of knots (degree + 1)
# k = 4

# # Fit B-spline to data points with clamped boundary conditions
# t, c, k = splrep(x_data, y_data, k=k-1)  # k-1 is the degree
# bspline = BSpline(t, c, k, extrapolate=False)

# # Generate points for plotting the B-spline
# x_plot = np.linspace(x_data[0], x_data[-1], 1000)
# y_plot = splev(x_plot, (t, c, k))

# # Plot the data points and the fitted B-spline
# plt.plot(x_data, y_data, 'o', label='Data Points')
# plt.plot(x_plot, y_plot, label='B-spline Fit')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.grid(True)
# plt.xlim(-1, 7)
# plt.show()


##############
#     def _protein_expression_estimation(self, A, max_card):
#         """Performs minimization of P2 in the paper

#         Args:
#             A (float[][]): A matrix
#             D (int): number of basis functions
#             max_card (int): maximal cardinality of M_t (number of non-coding RNAs found influencing the network)

#         Returns:
#             float[], float[]: np.arrays z and alpha
#         """
        
#         alpha = cp.Variable(self.D)  
#         constraints = []

#         K = np.zeros((self.D, self.D))

#         # Create K matrix (filled with integrated B-spline basis functions)
#         for j in range(self.D):
#             for k in range(self.D):
#                 w_j = [0 if i != j else 1 for i in range(self.D)]
#                 w_k = [0 if i != k else 1 for i in range(self.D)]
#                 bspl_second_order_der_j = BSpline(np.linspace(min(self.T), max(self.T), self.number_of_knots), w_j, self.k).derivative(nu=2)
#                 bspl_second_order_der_k = BSpline(np.linspace(min(self.T), max(self.T), self.number_of_knots), w_k, self.k).derivative(nu=2)
#                 phi_j = bspl_second_order_der_j(self.T)
#                 phi_k = bspl_second_order_der_k(self.T)
#                 K[j, k] = np.sum(phi_j*phi_k)



#         if max_card != 0:
#             z = cp.Variable(max_card, nonneg=True) 
#             # P2
#             # objective = cp.Minimize(cp.norm2(A @ cp.hstack((-1, z, alpha))) + cp.norm1(z)) # finish full expression
#             objective = cp.Minimize(cp.norm2(A @ cp.hstack((-1, z, alpha)))) # finish full expression
        
#             # Constraints
#             for t in range(len(A)):
#                 constraints.append(A[t, 1:max_card+1] @ z <= A[t, 0])
#         else:
#             # P2
#             # objective = cp.Minimize(cp.norm2(A @ cp.hstack((-1, alpha))) - alpha @ K @ alpha) # finish full expression # does not work somehow
#             objective = cp.Minimize(cp.norm2(A @ cp.hstack((-1, alpha)))) # finish full expression

#         problem = cp.Problem(objective, constraints)

#         problem.solve()

#         # Extract the optimal z and c values
#         if max_card != 0:
#             z_optimal = z.value
#         else:
#             z_optimal = None
#         alpha_optimal = alpha.value

#         return z_optimal, alpha_optimal

######################

# import numpy as np
# from scipy.optimize import minimize

# # Define your objective function 'A*x'
# def objective_function(x, A):
#    return np.sqrt(np.mean(np.dot(A, x)))

# # Define the constraint function 'A[3]*x[3] - A[1] < 0'
# def constraint_function(x, A):
#    return A[3] * x[2] - A[1]

# # Function to minimize
# def minimize_objective_with_constraint(A):
#    # Initial guess for 'x'
#    x0 = np.random.uniform(size=(len(A),))

#    # Define the constraint dictionary
#    constraint = {'type': 'ineq', 'fun': constraint_function, 'args': (A,)}

#    # Minimize the objective function subject to the constraint
#    result = minimize(objective_function, x0, args=(A,))

#    if result.success:
#       return result.x
#    else:
#       raise ValueError("Optimization failed.")

# # Example usage:
# A = np.random.randint(0, 100, size=(10, 10))
# optimized_x = minimize_objective_with_constraint(A)
# print("Optimized x:", optimized_x)
# print("Minimum value of Ax:", objective_function(optimized_x, A))

###########################

# import numpy as np
# from scipy.interpolate import BSpline
# import matplotlib.pyplot as plt

# def B(x, k, i, t):
#    if k == 0:
#       return 1.0 if t[i] <= x < t[i+1] else 0.0
#    if t[i+k] == t[i]:
#       c1 = 0.0
#    else:
#       c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
#    if t[i+k+1] == t[i+1]:
#       c2 = 0.0
#    else:
#       c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
#    return c1 + c2

# def B_der(x, k, i, t):
#    if k == 0:
#       return 1.0 if t[i] <= x < t[i+1] else 0.0
#    if t[i+k] == t[i]:
#       c1 = 0.0
#    else:
#       c1 = (k)/(t[i+k] - t[i]) * B(x, k-1, i, t)
#    if t[i+k+1] == t[i+1]:
#       c2 = 0.0
#    else:
#       c2 = (k)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
#    return c1 - c2

# def bspline_b(x, t, k):
#    n = len(t) - k - 1
#    assert (n >= k+1)
#    functions = []
#    for i in range(n):
#       functions.append(B(x, k, i, t))
#    return functions

# def get_individual_basis_functions(knots, degree, t):
#     num_basis = len(knots) - degree - 1
#     basis_functions = []
    
#     for i in range(num_basis):
#         # Set up the B-spline basis function
#         basis = np.zeros(num_basis)
#         basis[i] = 1
        
#         # Construct the B-spline object
#         spline = BSpline(knots, basis, degree, extrapolate=True)
        
#         # Evaluate the basis function at the given points
#         basis_values = spline(t)
        
#         basis_functions.append(spline)
    
#     return np.array(basis_functions)

# # Example usage
# knots = [0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1]
# degree = 3
# x = np.linspace(0, 1, 1000)  # Evaluation points
# num_basis = len(knots) - degree - 1

# for i in range(num_basis):
#    plt.plot(x, [B(j, degree, i, knots) for j in x])

# plt.xlim(0,1)
# plt.show()

# max_length_M_t = len(M_t[list(M_t.keys())[-1]]) # M_t(t_L) in the paper
# middle_vec = np.zeros((len(n1[0,1::]), len(x), max_length_M_t))

# for gene in range(1, len(n1[0,:])):
#     for t_idx, t in enumerate(x):
#         time_stamp = microRNA_expression(t, np.array(list(M_t.keys()))) # M_t.keys() are the time intervals
#         microRNA = []
#         for micro_gene in M_t[time_stamp]:
#             microRNA.append(n1[t_idx, micro_gene]*n1[t_idx, gene]) # is the micro_gene index correct? # MULTIPLY WITH GENE EXPRESSION OF GENE X_i
        
#         if len(microRNA) < max_length_M_t:
#             microRNA = microRNA + [0 for _ in range(max_length_M_t - len(microRNA))]
#         middle_vec[gene-1, t_idx] = microRNA

# # A matrix (equation 10) (for all genes) 
# A = np.zeros((len(n1[0,1::]), len(x), max_length_M_t + D + 1)) # (number of genes, number of time points, max_length_M_t + D + 1). Of note, per network, you always have the same number for max_length_M_t

# for gene in range(1, len(n1[0,:])):
#     x_i = n1[:, gene]
#     A[gene - 1, :, 0] = x_i.T
            
#     for t in range(len(x)):
#         A[gene - 1, t, 1::] = np.concatenate((middle_vec[gene -1, t], b[gene - 1, t, :]))    

# # build for loop to create y_i(t)
# y_i_t = np.zeros((len(n1), len(n1[0]))) # size = (time_points, number_of_genes)

# z, alpha = protein_expression_estimation(A[0], D, max_length_M_t)
# print(z)
# print(alpha)

# PLOTTING ALL 5 NETWORKS

# fig, axs = plt.subplots(5, 1, sharex=True, sharey=True, figsize=(18, 15))

# for j in range(5):
#     for i in range(10):
#         axs[j].plot(networks[j][:, 0], networks[j][:,i+1], '-o')
#     axs[j].set_title("network " + str(j+1))
#     axs[j].set_ylim([0, 1])
#     axs[j].grid()

# fig.tight_layout()
# plt.show()

# plt.title("BSpline curve fitting")
# plt.plot(x, y, 'ro', label="original")
# plt.plot(x_new, y_fit, '-c', label="B-spline")
# plt.legend(loc='best', fancybox=True, shadow=True)
# plt.grid()
# plt.show() 

# def get_bspline_vector(T, number_of_knots, k):
   #  """Generates 2 vectors that contain the basis b-spline function and the derivative of the b-spline basis function, respectively. They each have length
   #  D (number of basis functions) and every entry in a vector thus contains the i'th basis function of weighted sum that the SciPy BSpline normally returns. 
   #  Individual Bspline basis functions are non-zero on certain domains described by the knots of the Bspline (https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-basis.html)
   #  More specifically, the i'th Bspline basis function is non zero on the interval [t_j, t_{j+k+1}) where t_j describes the j'th knot and k describes the degree of
   #  the basis function. Of note, we do not actually use the gene expression data itself to construct these B-spline functions, rather the time points given in the `networks` matrix.

   #  Args:
   #      T (float[]): array of measured time points
   #      number_of_knots (int): number of knots to be used --> uniform knot distribution is used 
   #      k (int): degree of B-spline basis functions

   #  Returns:
   #      ([Bspline()], [Bspline().derivative()]): B-spline basis functions, and derivative functions
   #  """

    # bspl_vec = []
    # bspl_der_vec = []

    # x = networks[:,0]
    # knots = np.linspace(min(T), max(T), number_of_knots)
    # D = number_of_knots - k - 1
    # bspl = BSpline(t=knots, k=k, c=[1 for i in range(D)])
    # bspl_derivative = bspl.derivative() # first order derivative
    # non_zero_intervals_basis_functions = [[knots[i] for i in range(j, j+k+1+1)] for j in range(D)] # non zero interval for basis function j is [t_j, t_{j+k+1}) but since range function excludes the last value you add +1

    # for i in range(D):
    #     # bspl_vec.append(bspl.basis_element(non_zero_intervals_basis_functions[i], extrapolate=False))
    #     # bspl_der_vec.append(bspl_derivative.basis_element(non_zero_intervals_basis_functions[i], extrapolate=False))
    #     bspl_vec.append(bspl.basis_element(non_zero_intervals_basis_functions[i], extrapolate=True))
    #     bspl_der_vec.append(bspl_derivative.basis_element(non_zero_intervals_basis_functions[i], extrapolate=True))
    
    # return bspl_vec, bspl_der_vec


############################

        # for g in range(self.number_of_genes):
        #     max_length_M_t = len(M_t[list(M_t.keys())[-1]]) # M_t(t_L) in the paper
        #     middle_vec = np.zeros((len(self.T), max_length_M_t))
            
        #     for t_idx, t in enumerate(T):
        #         time_stamp = self._microRNA_expression(t, np.array(list(M_t.keys()))) # M_t.keys() are the time intervals
        #         microRNA = []
        #         for micro_gene in M_t[time_stamp]:
        #             # microRNA.append(self.rna_expression[t_idx, micro_gene]*x_fit[g](t)) # is the micro_gene index correct? if M_t detects gene 0, then this indexing is incorrect since the 0th column represents time # MULTIPLY WITH GENE EXPRESSION OF GENE X_i
        #             microRNA.append(self.rna_expression[t_idx, micro_gene]*self.rna_expression[t_idx, g]) # is the micro_gene index correct? if M_t detects gene 0, then this indexing is incorrect since the 0th column represents time # MULTIPLY WITH GENE EXPRESSION OF GENE X_i

        #         if len(microRNA) < max_length_M_t:
        #             microRNA = microRNA + [0 for _ in range(max_length_M_t - len(microRNA))]
        #         middle_vec[t_idx, :] = microRNA

        #     # A matrix (equation 10) (for all genes) 
        #     A = np.zeros((len(T), max_length_M_t + self.D + 1)) # (number of time points, max_length_M_t + D + 1). Of note, per network, you always have the same number for max_length_M_t
        # x_i = np.zeros(len(T))
        # for i, t in enumerate(T):
        #     x_i[i] = x_fit[g](t)
        # # A[:, 0] = x_i.T
        

        # for t_idx in range(len(T)): 
        #     A[t_idx, 1::] = np.concatenate((middle_vec[t_idx], b[g, t_idx, :]))  
############################

        #     x_i = self.rna_expression[:, g]
        #     # x_i = np.zeros(len(T))
        #     # for i, t in enumerate(T):
        #     #     x_i[i] = x_fit[g](t)
        #     # A[:, 0] = x_i.T
            

        #     for t_idx in range(len(T)): 
        #         A[t_idx, 1::] = np.concatenate((middle_vec[t_idx], b[g, t_idx, :]))    

        # b = []
        # for g in range(self.number_of_genes):
        #     b_i = []
        #     for t in T:
        #         b_i_t = []
        #         for d in range(self.D):
        #             lmda_PROT = protein_degradation_rate[g]
        #             b_i_t.append(lmda_PROT*Basis_b_spline(t, k, d, self.knots) + B_der(t, k, d, self.knots))
        #         b_i.append(b_i_t)
        #     b.append(b_i)

        # b = np.array(b) # size: (number_of_genes, time points, D)

        # middle vector in A matrix (equation 10)

        # # TEST CODE
        # for i in (600, 800, 1000):
        #     if i == 600:
        #         M_t[i] = set([2,4])
        #     elif i > 600:
        #         M_t[i] = set([2,4, 6])

        # print(M_t)

##############################

# def y_i(t, bspl_vec, alpha):
#     """Implements equation 8 after calculating alpha's of interest.

#     Args:
#         t (float): time point at which to evaluate the protein expression level of gene of interest
#         bspl_vec (Bspline[]): list of Bspline basis functions (specific to a gene)
#         alpha (float[]): weights of Bspline basis functions

#     Returns:
#         float: protein expression of gene i at time point t, i.e. y_i(t)
#     """
#     phi_t = np.array([bspl_vec[i](t) for i in range(len(bspl_vec))]) 
#     y = np.dot(alpha, phi_t)
#     return y

# def Basis_b_spline(x, k, i, t):
#     """Recursive version of the ith B-spline basis function. Copied from https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html

#     Args:
#         x (float): (time) point at which to evaluate the basis function
#         k (int): degree of basis function
#         i (int): index of basis function in sum(c[i] * B(x, k, i, t)) 
#         t (float[]): knot vector

#     Returns:
#         float: evaluation of the ith B-spline function at point x
#     """
#     if k == 0:
#         return 1.0 if t[i] <= x < t[i+1] else 0.0
#     if t[i+k] == t[i]:
#         c1 = 0.0
#     else:
#         c1 = (x - t[i])/(t[i+k] - t[i]) * Basis_b_spline(x, k-1, i, t)
#     if t[i+k+1] == t[i+1]:
#         c2 = 0.0
#     else:
#         c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * Basis_b_spline(x, k-1, i+1, t)
#     return c1 + c2

# def B_der(x, k, i, t):
#     """Recursive version of the derivative ith B-spline basis function. Derived from https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-derv.html

#     Args:
#         x (float): (time) point at which to evaluate the basis function
#         k (int): degree of basis function
#         i (int): index of basis function in sum(c[i] * B(x, k, i, t)) 
#         t (float[]): knot vector

#     Returns:
#         float: evaluation of the derivative of the ith B-spline function at point x
#     """
#     if k == 0:
#         return 1.0 if t[i] <= x < t[i+1] else 0.0
#     if t[i+k] == t[i]:
#         c1 = 0.0
#     else:
#         c1 = (k)/(t[i+k] - t[i]) * Basis_b_spline(x, k-1, i, t)
#     if t[i+k+1] == t[i+1]:
#         c2 = 0.0
#     else:
#         c2 = (k)/(t[i+k+1] - t[i+1]) * Basis_b_spline(x, k-1, i+1, t)
#     return c1 - c2

# def b_vec(t, bspl_vec, bspl_derivative_vec, lmbda_PROT):
#     """Implements b_i(t) expression given on page 1097 at the top left paragraph. 

#     Args:
#         t (float): time point at which to evaluate the expression
#         bspl_vec (Bspline[]): list of Bspline basis functions
#         bspl_derivative_vec (Bspline[]): list of Bspline basis functions derivatives 
#         lmbda_PROT (float): gene specific protein degradation constant 

#     Returns:
#         _type_: _description_
#     """
#     phi_t = np.array([bspl_vec[i](t) for i in range(len(bspl_vec))])
#     phi_der_t = np.array([bspl_derivative_vec[i](t) for i in range(len(bspl_derivative_vec))])
#     b = lmbda_PROT*phi_t + phi_der_t

#     return b

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
