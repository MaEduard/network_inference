import matplotlib.pyplot as plt
import numpy as np
from read_data import *
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from b_spline_fitting import x_fit_bspline, get_bspline_vector, b_vec
import cvxpy as cp

def microRNA_expression(t, T):
    """Finds time interval, described by its upperbound, to which t belongs to generate all microRNAs expressed up till t. 

    Args:
        t (float): time point
        T (float[]): list of time point intervals

    Returns:
        float: time point interval to which t belongs
    """
    idx = T[T > t]
    if len(idx) == 0:
        return T[len(T)-1]
    return idx[0]

def protein_expression_estimation(A, D, max_card):
    """Performs minimization of P2 in the paper

    Args:
        A (float[][]): A matrix
        D (int): number of basis functions
        max_card (int): maximal cardinality of M_t (number of non-coding RNAs found influencing the network)

    Returns:
        float[], float[]: np.arrays z and alpha
    """
    z = cp.Variable(max_card, nonneg=True)
    alpha = cp.Variable(D)  

    # P2
    objective = cp.Minimize(cp.norm2(A @ cp.hstack((-1, z, alpha))) + cp.norm1(z)) # finish full expression

    # Constraints
    constraints = []
    for t in range(len(A)):
        constraints.append(A[t, 1:max_card+1] @ z <= A[t, 0])

    problem = cp.Problem(objective, constraints)

    problem.solve()

    # Extract the optimal z and c values
    z_optimal = z.value
    alpha_optimal = alpha.value

    return z_optimal, alpha_optimal

def main(n1, lmbda, k, number_of_knots, show_plot=False):
    # dummy data to compare with paper --> remove nets etc. later
    x = n1[:,0]
    x_new = np.linspace(min(x), max(x), 1000)
    nets = [i for i in range(len(n1[0,:])) if i != 0] # except 0 since those are the time points

    x_fit = x_fit_bspline(n1, lmbda=lmbda, k=k, number_of_knots=number_of_knots) # also returns fitted on time data points (first column) but can be ignored to make indexing more intuitive

    # Detection of perturbed genes
    D = number_of_knots - k - 1 # number of basis functions used
    R = 6 # R >= D (according to paper) where R is the number of intervals you'd like to perform the gene expression detection on.
    assert R >= D, "R must >= D"
    T = 0.15 # threshold for detection, taken from paper (according to paper T \epsilon [0.15, 0.20])
    t_intervals = np.linspace(0, max(x), R)
    t_int = t_intervals[1]

    # G_t & M_t definition as in the paper 
    G_t = {i: set() for i in t_intervals[1::]} # The set of all indices that correspond to protein encoding genes that were detected having changed gene expression. Start with interval [0,x) which is indicated as key x. Then interval [x, 2*x) etc.
    M_t = {i: set() for i in t_intervals[1::]} # The set of all indices that correspond to miRNA-encoding genes that were detected having changed gene expression. Start with interval [0,x) which is indicated as key x. Then interval [x, 2*x) etc.
    not_detected = [i for i in nets] # list of non detected genes (deep copy)

    # detect per interval whether a gene is influenced by perturbed gene. If difference in gene expresssion between beginning (t-t_int) and end (t) of interval > T, add gene to set of genes being pertured from time point 
    for j, t in enumerate(t_intervals[1:len(t_intervals)]):
        to_be_removed = []
        for gene_i in not_detected:
            if abs(x_fit[gene_i](t-t_int) - x_fit[gene_i](t)) > T:
                to_be_removed.append(gene_i)
                for tt in t_intervals[j+1:len(t_intervals)]:
                    G_t[tt].add(gene_i) # ADAPT HERE TO DISTINGUISH BETWEEN M_t and G_t. If gene is miRNA ecoding gene, M_t[t+1].add(i) else G_t[t+1].add(i)

        for gene_j in to_be_removed:
            not_detected.remove(gene_j)
            
    # Plotting gene expression
    if show_plot:
        # plotting settings
        colors = ['b', 'r', 'lime', 'magenta', 'black', 'sienna', 'darkolivegreen', 'turquoise', 'hotpink', 'goldenrod']
        markers = ["o", "d", "v", "s", "*", "^", "o", "d", "v", "s", "*", "^"]

        for i in range(len(nets)):
            y_pred = x_fit[nets[i]](x_new)
            plt.plot(x, n1[:,nets[i]], label=f"gene {nets[i]}", color=colors[i % len(colors)], linewidth=1.0)
            plt.plot(x_new, y_pred, '--', color=colors[i % len(colors)], linewidth=1.0)

        # plotting
        plt.title(f"Network 1 of insilico_size10_2 for lmda = {lmbda}, k = {k}, knots = {number_of_knots}")
        plt.ylabel("Expresssion levels")
        plt.xlabel("Time")
        plt.ylim([0, 1.1])
        plt.legend(loc='upper right', fancybox = True, shadow=True)
        plt.grid()
        plt.show()

    # protein expression estimation

    # b vector in A matrix (equation 10)
    bspl_vec, bspl_derivative_vec = get_bspline_vector(n1, number_of_knots, k)
    protein_degradation_rate = np.array([1 for _ in range(len(n1[0,1::]))]) # make all 1 (simplification) 
    # protein_translation_rate = np.array([1 for _ in range(len(n1[0,1::]))]) # make all 1 (simplification)

    b = []
    for indx in range(len(n1[0,1::])):
        b_i = []
        for t in x:
            lmbda_PROT = protein_degradation_rate[indx]
            b_i_t = b_vec(t, bspl_vec=bspl_vec, bspl_derivative_vec=bspl_derivative_vec, lmbda_PROT=lmbda_PROT)
            b_i.append(b_i_t)
        b.append(b_i)

    b = np.array(b) # size: (10, 21, 4) --> (number_of_genes, time points, D) 

    # middle vector in A matrix (equation 10)

    # TEST CODE
    for i in (600, 800, 1000):
        if i == 600:
            M_t[i] = set([2,4])
        elif i > 600:
            M_t[i] = set([2,4, 6])

    # print(M_t)

    max_length_M_t = len(M_t[list(M_t.keys())[-1]]) # M_t(t_L) in the paper
    middle_vec = np.zeros((len(n1[0,1::]), len(x), max_length_M_t))
    
    for gene in range(1, len(n1[0,:])):
        for t_idx, t in enumerate(x):
            time_stamp = microRNA_expression(t, np.array(list(M_t.keys()))) # M_t.keys() are the time intervals
            microRNA = []
            for micro_gene in M_t[time_stamp]:
                microRNA.append(n1[t_idx, micro_gene]*n1[t_idx, gene]) # is the micro_gene index correct? # MULTIPLY WITH GENE EXPRESSION OF GENE X_i
            
            if len(microRNA) < max_length_M_t:
                microRNA = microRNA + [0 for _ in range(max_length_M_t - len(microRNA))]
            middle_vec[gene-1, t_idx] = microRNA

    # A matrix (equation 10) (for all genes) 
    A = np.zeros((len(n1[0,1::]), len(x), max_length_M_t + D + 1)) # (number of genes, number of time points, max_length_M_t + D + 1). Of note, per network, you always have the same number for max_length_M_t

    for gene in range(1, len(n1[0,:])):
        x_i = n1[:, gene]
        A[gene - 1, :, 0] = x_i.T
                
        for t in range(len(x)):
            A[gene - 1, t, 1::] = np.concatenate((middle_vec[gene -1, t], b[gene - 1, t, :]))    
    
    z, alpha = protein_expression_estimation(A[0], D, max_length_M_t)
    print(z)
    print(alpha)

if __name__ == "__main__":
    # Load data
    path_name = os.path.join("data", "DREAM4", "DREAM4_InSilico_Size10", "insilico_size10_2" ,"insilico_size10_2_timeseries.tsv")
    n1, n2, n3, n4, n5 = read_dream_time_series(path_name, 21) # returns np.arrays
    networks = [n1, n2, n3, n4, n5]
    # Modeling and Estimation of Gene Expression
    lmbda = 1e4
    k = 3 # degree of B-spline
    number_of_knots = 8
    show_plot = False
    main(n1, lmbda, k, number_of_knots, show_plot)