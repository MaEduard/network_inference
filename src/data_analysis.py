import matplotlib.pyplot as plt
import numpy as np
from read_data import *
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from b_spline_fitting import x_fit_bspline

# Load data
path_name = os.path.join("data", "DREAM4", "DREAM4_InSilico_Size10", "insilico_size10_2" ,"insilico_size10_2_timeseries.tsv")
n1, n2, n3, n4, n5 = read_dream_time_series(path_name, 21)
networks = [n1, n2, n3, n4, n5]

# dummy data to compare with paper --> remove nets etc. later
x = n2[:,0]
x_new = np.linspace(min(x), max(x), 1000)
nets = [i for i in range(len(n2[0,:])) if i != 0] # except 0 since those are the time points

# Modeling and Estimation of Gene Expression
lmbda = 1e4
k = 3
number_of_knots = 9
x_fit = x_fit_bspline(n2, lmbda=lmbda, k=k, number_of_knots=number_of_knots) # also returns fitted on time data points (first column) but can be ignored to make indexing more intuitive

# Detection of perturbed genes
D = number_of_knots - k -1 # number of basis functions used
R = 6 # R >= D (according to paper)
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
        
print(G_t)

### PLOTTING ###

# plotting settings

colors = ['b', 'r', 'lime', 'magenta', 'black', 'sienna', 'darkolivegreen', 'turquoise', 'hotpink', 'goldenrod']
markers = ["o", "d", "v", "s", "*", "^", "o", "d", "v", "s", "*", "^"]

for i in range(len(nets)):
    y_pred = x_fit[nets[i]](x_new)
    plt.plot(x, n2[:,nets[i]], label=f"gene {nets[i]}", color=colors[i % len(colors)], linewidth=1.0)
    plt.plot(x_new, y_pred, '--', color=colors[i % len(colors)], linewidth=1.0)

# plotting
plt.title(f"Network 1 of insilico_size10_2 for lmda = {lmbda}, k = {k}, knots = {number_of_knots}")
plt.ylabel("Expresssion levels")
plt.xlabel("Time")
plt.ylim([0, 1.1])
plt.legend(loc='upper right', fancybox = True, shadow=True)
plt.grid()
plt.show()
