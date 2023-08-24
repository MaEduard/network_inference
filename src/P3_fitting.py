import numpy as np
from scipy.optimize import minimize, Bounds

def objective(x, P, gamma1, gamma2):
    n = len(x) // 2
    a = x[:n]
    b = x[n:]
    # return np.linalg.norm(P.dot(x), ord=2) + gamma1*np.linalg.norm(a, ord=1) + gamma2*np.linalg.norm(b, ord=1)
    # return np.sum(P.dot(x)**2) + gamma1*np.linalg.norm(b, ord=1) 
    return  np.linalg.norm(P.dot(x), ord=2)  + gamma1*np.linalg.norm(b, ord=1) + gamma1*np.linalg.norm(a, ord=1)

def constraint1(x):
    n = len(x) // 2
    return x[n] - 1  # Constraint: The first entry in b should be 1

def constraint2(x):
    n = len(x) // 2
    a = x[:n]
    b = x[n:]
    return b - a # Constraint: a <= b

def constraint3(x):
    n = len(x) // 2
    a = x[:n]
    return a # Constraint: 0 <= a

def P3(P):

    n = len(P[0]) // 2
    # Initial guess for optimization variables (a and b concatenated)
    w0 = np.ones(2 * n)

    # Define the bounds for variables a and b
    # bounds = [(0, None)] * n + [(1, None)] + [(0, None)] * (n-1)
    bounds = Bounds(lb=[0]*2*n, ub=[np.inf]*2*n)
    # Define the constraints
    constraints = [
        {'type': 'eq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2},
        {'type': 'ineq', 'fun': constraint3}
    ]

    ineq_cons1 = {'type': 'ineq',
                'fun' : lambda x: x[n:] -  x[:n]
                }
    ineq_cons2 = {'type': 'ineq',
                'fun' : lambda x: x[:n]
                }
    eq_cons = {'type': 'eq',
            'fun' : lambda x: np.array([x[n] - 1])
            }
    # {'type': 'eq', 'fun': constraint1}
    ground_truth = np.array([0.1, 0, 0.1, 0, 0, 0, 0, 1, 0, 0.1, 0.1, 0, 0, 0])
    
    optimal_gamma1 = 0
    optimal_gamma2 = 0
    optimal = 100
    # gamma1 = np.logspace(-10,2,13)
    # gamma2 = np.logspace(-10,2,13)
    # gamma3 = np.logspace(-10,2,13)
    # gamma2 = [0]
    # gamma1 = [0.001]
    # gamma1 = np.logspace(-10,2,13)
    # gamma2 = np.logspace(-10,2,13)
    gamma1 = [0.1]
    gamma2 = [0]
    for i in gamma1:
        for j in gamma2:
            result = minimize(objective, w0, bounds=bounds, constraints=[eq_cons, ineq_cons1, ineq_cons2], args=(P, i, j))
            # mse = (np.abs(result.x - ground_truth).mean())
            print(i)
            print(j)
            print(result.success)
            print(result.x[:n].round(3))
            print(result.x[n:].round(3))
            # if mse < optimal:
            #     optimal = mse
            #     optimal_gamma1 = i
            #     optimal_gamma2 = j

            # if result.success == False:
            #     print(result)
    
    # print(optimal_gamma1)
    # print(optimal_gamma2)
    # print(result.x.round(3))
    # print(optimal)
    return result.x


############################################

# import cvxpy as cp
# import numpy as np

# def P3(P):
#     n = len(P[0])
#     x = cp.Variable(n)
#     gamma1 = np.logspace(-10, 3, 14)
#     gamma2 = np.logspace(-10, 3, 14)

#     for i in gamma1:
#         for j in gamma2:
#             objective = cp.Minimize(cp.sum_squares(P @ x) + i*cp.norm1(x[(n//2):]) + (j/2)*cp.norm2(x[(n//2):])**2)  # Adding L1 norm regularization
#             constraints = [0 <= x, x[:(n//2)] <= x[(n//2):], x[n//2] == 1]
#             prob = cp.Problem(objective, constraints)

#             # The optimal objective value is returned by `prob.solve()`.
#             result = prob.solve()
#             # The optimal value for x is stored in `x.value`.
#             print(i)
#             print(j)
#             print(x.value[:(n//2)].round(3))
#             print(x.value[(n//2):].round(3))

#############

# from scipy.optimize import minimize
# import numpy as np

# def loss_function(weights, gene_i, T, x_fit, p_vector, lmbda_RNA):
#     """Not yet regularized

#     Args:
#         gene_i (_type_): _description_
#         T (_type_): _description_
#         x_fit (_type_): _description_
#         p_vector (_type_): _description_
#         b_vecs (_type_): _description_
#         lmbda_RNA (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     omega = 0
#     # weights[int(len(weights)/2)] = 1 # added in constraints as well but not too sure if it works there. 

#     for t_indx, t in enumerate(T):
#             #   - 1000000*np.linalg.norm(weights[int(len(weights)/2):len(weights)], ord=1)
#         omega += -x_fit[gene_i].derivative()(t) * (p_vector[gene_i][t_indx].dot(weights[int(len(weights)/2):len(weights)])) - lmbda_RNA * x_fit[gene_i](t)*(p_vector[gene_i][t_indx].dot(weights[int(len(weights)/2):len(weights)])) + (p_vector[gene_i][t_indx].dot(weights[0:int(len(weights)/2)])) + 1000000*np.linalg.norm(weights[int(len(weights)/2):len(weights)], ord=1) 
#     return omega

# def con_1(weights):
#     """b(0) = 1
#     """
#     return weights[int(len(weights)/2)] - 1

# def con_2(weights):
#     """a_i >= 0
#     """
#     return weights[0:int(len(weights)/2)]

# def con_3(weights):
#     """b_i >= a_i
#     """
#     return weights[int(len(weights)/2):len(weights)] - weights[0:int(len(weights)/2)] # b_i >= a_i --> b_i - a_i >= 0

# def P3(x_fit, T, lmbda_RNA, p_vector, number_of_genes):

#     # Initialize the weights
#     a_b_vecs = np.zeros((number_of_genes, 2*len(p_vector[0][0]))) # size = (number of genes, N_t_L)

#     cons = [{'type':'eq', 'fun': con_1},
#         {'type':'ineq', 'fun': con_2},
#         {'type':'ineq', 'fun': con_3}]
#     bs = [(0, None)] * len(p_vector[0][0]) + [(1, None)] + [(0, None)] * (len(p_vector[0][0])-1)
#     # Minimize the loss function
#     for gene_i in range(number_of_genes):
#         weights = np.random.uniform(size=2*len(p_vector[0][0]))
#         # weights[int(len(weights)/2)] = 1
#         result = minimize(loss_function, weights, args=(gene_i, T, x_fit, p_vector, lmbda_RNA[gene_i]), constraints=cons, bounds=bs, method="SLSQP")
#         # Get the optimal weights and calculate the predicted values
#         w = result.x
#         print(result)
#         a_b_vecs[gene_i] = w

#     return a_b_vecs