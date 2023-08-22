import matplotlib.pyplot as plt
import numpy as np
from read_data import *
from scipy.interpolate import BSpline
from b_spline_fitting import x_fit_bspline, fit_bspline_protein_expression
from plotting import time_vs_expression
from itertools import chain, combinations
from P3_fitting import P3
from sklearn.preprocessing import normalize
import pandas as pd


class NetworkInference():

    def __init__(self, T, rna_expression, lmbda, R, knots, k=3, detection_threshold=0.15) -> None:
        self._T = T
        self._rna_expression = rna_expression
        self._lmbda = lmbda
        self._knots = knots
        self._R = R
        self._k = k
        self._D = len(knots) - k - 1
        self._detection_threshold = detection_threshold 
        self._number_of_genes = len(rna_expression[0])
        if self.R < self.D:
            raise Exception('The following must hold: R >= D. Currently, R = {} and D = {}'.format(R, self.D))

    @property
    def T(self):
        return self._T.copy()
    
    @property
    def rna_expression(self):
        return self._rna_expression.copy()

    @property
    def lmbda(self):
        return self._lmbda
    
    @property
    def knots(self):
        return self._knots
    
    @property
    def number_of_knots(self):
        return len(self.knots)

    @property
    def R(self):
        return self._R
    
    @property
    def k(self):
        return self._k
    
    @property
    def D(self):
        return self._D
    
    @property
    def detection_threshold(self):
        return self._detection_threshold

    @property
    def number_of_genes(self):
        return self._number_of_genes
    
    def _microRNA_expression(cls, t, microRNA_T):
        """Finds time interval, described by its upperbound, to which t belongs to generate all microRNAs expressed up till t. 

        Args:
            t (float): time point
            T (float[]): list of time point intervals

        Returns:
            float: time point interval to which t belongs
        """
        idx = microRNA_T[microRNA_T > t]
        if len(idx) == 0:
            return microRNA_T[len(microRNA_T)-1]
        return idx[0]
    

    def _powerSet(self, iterable):
        """Computes the powerset of a given iterable of numbers. Filters out any set bigger than size 3 as the paper restricts
        genes to be regulated by at max 2 proteins to reduce the complexity of the model.

        Example: [1, 2, 3] --> [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3)]

        Args:
            iterable (int[]): list of genes

        Returns:
            tuple[]: list of possible gene combinations that can regulate a gene
        """
        s = list(iterable)
        powerset = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
        shrinked_powerset = [item for item in powerset if len(item) < 3]
        return shrinked_powerset
    
    def infer(self, show_plot=False):
        
        # P1
        TT = np.linspace(min(self.T), max(self.T), 1000) # time points for plotting
        x_fit = x_fit_bspline(self.T, self.rna_expression, self.knots, self.lmbda, self.k) # also returns fitted on time data points (first column) but can be ignored to make indexing more intuitive
        t_intervals = np.linspace(0, max(self.T), self.R + 1) # is it best to do this + 1? Otherwise you 1 interval short according to the paper
        t_interval_size = t_intervals[1]

        # G_t & M_t definition as in the paper 
        G_t = {i: set() for i in t_intervals[1::]} # The set of all indices that correspond to protein encoding genes that were detected having changed gene expression. Start with interval [0,x) which is indicated as key x. Then interval [x, 2*x) etc.
        M_t = {i: set() for i in t_intervals[1::]} # The set of all indices that correspond to miRNA-encoding genes that were detected having changed gene expression. Start with interval [0,x) which is indicated as key x. Then interval [x, 2*x) etc.
        not_detected = [i for i in range(self.number_of_genes)] # list of non detected genes

        # detect per interval whether a gene is influenced by perturbed gene. If difference in gene expresssion between beginning (t-t_int) and end (t) of interval > T, add gene to set of genes being pertured from time point 
        for j, t in enumerate(t_intervals[1:len(t_intervals)]):
            to_be_removed = []
            for gene_i in not_detected:
                if abs(x_fit[gene_i](t-t_interval_size) - x_fit[gene_i](t)) > self.detection_threshold:
                    to_be_removed.append(gene_i)
                    for tt in t_intervals[j+1:len(t_intervals)]:
                        G_t[tt].add(gene_i) # ADAPT HERE TO DISTINGUISH BETWEEN M_t and G_t. If gene is miRNA ecoding gene, M_t[t+1].add(i) else G_t[t+1].add(i)
                    
            for gene_j in to_be_removed:
                not_detected.remove(gene_j)

        # Plotting gene expression
        if show_plot:
            time_vs_expression(self.T, TT, x_fit, self.number_of_genes, self.rna_expression, knots, lmbda, k)

        # P2
        protein_degradation_rate = np.array([0.5 for _ in range(self.number_of_genes)]) # make all 1 (simplification) 
        # protein_translation_rate = np.array([1 for _ in range(self.number_of_genes)]) # make all 1 (simplification)
        protein_translation_rate = np.array([1,2,1])

        alpha_i = np.zeros((self.number_of_genes, self.D)) # size = (number of genes, number_of_basis_functions)

        for g in range(self.number_of_genes):
            x_i = x_fit[g](self.T)
            alpha_i[g, :] = fit_bspline_protein_expression(protein_translation_rate[g], protein_degradation_rate[g], self.T, self.knots, self.k, self.D, x_i)
        if show_plot:
            time_vs_expression(self.T, TT, x_fit, self.number_of_genes, self.rna_expression, self.knots, self.lmbda, self.k, False, alpha_i)

        # P3
        N_t_L = len(self._powerSet(list(G_t.values())[-1])) # largest set of regulators (out of steady-state genes) found. I.e. N(t_L) in the paper.
        print(G_t)
        p_vector = np.zeros((self.number_of_genes, len(self.T), 2*N_t_L)) # matrix holding p vector as described in equation 11 
        for g in range(self.number_of_genes):
            for t_idx, t in enumerate(self.T):
                interval = self._microRNA_expression(t, np.array(list(G_t.keys()))) # finds interval to which time point t belongs 
                protein_regulators = self._powerSet(G_t[interval]) # point of optimalization is possible
                for reg_idx, regulators in enumerate(protein_regulators): 
                    y_k = 1 
                    for regulator in regulators: 
                        y_k *=  BSpline(knots, alpha_i[regulator], k=self.k)(t) 
                    p_vector[g, t_idx, reg_idx] = y_k 
                    if y_k < 0: 
                        print(f"t = {t}, regulators = {regulators}") 
                        print(BSpline(knots, alpha_i[regulator], k=self.k)(t))


        # Multiply with constant C(t) = lambda_i^RNA*x_i(t) + dx/dt(t)
        lmbda_RNA = [0.1, 0.1, 0.1] # taken from the paper
        delta_t = self.T[1] - self.T[0]
        for g in range(self.number_of_genes):
            for t_idx, t in enumerate(self.T):
                c_t = lmbda_RNA[g] * x_fit[g](t) + ((x_fit[g](t + delta_t) - x_fit[g](t))/delta_t)
                # c_t = lmbda_RNA[g] * x_fit[g](t) + x_fit[g].derivative()(t)
                c_t_p = p_vector[g, t_idx, :N_t_L] * -c_t 
                p_vector[g, t_idx, N_t_L:] = c_t_p

        df = pd.DataFrame(p_vector[2])
        # df.to_clipboard(index=False,header=False)

        print(G_t)

        # coef1 = P3(p_vector[0])
        # coef2 = P3(p_vector[1])
        coef3 = P3(p_vector[2])

if __name__ == "__main__":
    # Load data DREAM CHALLENGE

    # path_name = os.p  ath.join("data", "DREAM4", "DREAM4_InSilico_Size10", "insilico_size10_2" ,"insilico_size10_2_timeseries.tsv")
    # n1, n2, n3, n4, n5 = read_dream_time_series(path_name, 21) # returns np.arrays
    # T = n1[:,0]
    # rna_expression = n1[:, 1::]
    
    # Load data EXPERIMENT 4.1
    path_name = os.path.join("data", "experiment_4_1", "experiment_4_1.tsv")
    data = read_experiment_41(path_name)
    T = data[:,0]
    rna_expression = data[:, 1::]
    rna_expression_normalized_per_gene_norm = np.empty_like(rna_expression)
    for column in range(len(rna_expression[0])):
        # rna_expression_normalized_per_gene[:, column] = (rna_expression[:, column] - min(rna_expression[:, column])) / (max(rna_expression[:, column]) - min(rna_expression[:, column]))
        rna_expression_normalized_per_gene_norm[:, column] = rna_expression[:, column] / np.linalg.norm(rna_expression[:, column], ord=2)

    rna_expression_normalized_all_genes = np.empty_like(rna_expression)
    min_all = np.min(rna_expression)
    max_all = np.max(rna_expression)
    for column in range(len(rna_expression[0])):
        rna_expression_normalized_all_genes[:, column] = (rna_expression[:, column] - min_all) / (max_all - min_all)
        
    rna_expression_normalized_per_gene = np.empty_like(rna_expression)
    for column in range(len(rna_expression[0])):
        min_per_gene = np.min(rna_expression[:, column])
        max_per_gene = np.max(rna_expression[:, column])
        rna_expression_normalized_per_gene[:, column] = (rna_expression[:, column] - min_per_gene) / (max_per_gene - min_per_gene)

    lmbda = 15
    k = 3 # degree of B-spline  
    number_of_knots = 10
    R = 6
    detection_threshold = 0.02
    show_plot = True
    # knots = np.linspace(min(T), max(T), number_of_knots)
    knots = np.array([0, 2, 4, 5, 10, 15, 20, 30, 50])
    knots = np.linspace(0,50, 10)
    # clamped_knots = np.array((k+1)*[min(T)] + np.floor(np.linspace(T[0], T[-1], number_of_knots - 2*k)).tolist() + (k+1)*[max(T)])
    # print(clamped_knots)
    network = NetworkInference(T, rna_expression_normalized_all_genes, lmbda, R, knots, k, detection_threshold)
    network.infer(show_plot)