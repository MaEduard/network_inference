import numpy as np
from read_data import *
from scipy.interpolate import BSpline
from b_spline_fitting import x_fit_bspline, fit_bspline_protein_expression
from plotting import time_vs_expression
from itertools import chain, combinations
from P3_fitting import P3
import os

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
    
    def _interval_detection(cls, t, intervals):
        """Finds time interval, described by its upperbound, to which t belongs.
        Args:
            t (float): time point
            T (float[]): list of time point intervals

        Returns:
            float: time point interval to which t belongs
        """
        idx = intervals[intervals > t]
        if len(idx) == 0:
            return intervals[len(intervals)-1]
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
        
        ### P1 ###
        TT = np.linspace(min(self.T), max(self.T), 1000) # time points for plotting
        x_fit = x_fit_bspline(self.T, self.rna_expression, self.knots, self.lmbda, self.k) # solving minimization problem described by P1
        t_intervals = np.linspace(0, max(self.T), self.R + 1) # create interval boundaries 
        t_interval_size = t_intervals[1]

        # G_t & M_t definition as in the paper 
        G_t = {i: set() for i in t_intervals[1::]} # The set of all indices that correspond to protein encoding genes that were detected having changed gene expression significantly. Start with interval [0,x) which is indicated as key x. Then interval [x, 2*x) etc.
        M_t = {i: set() for i in t_intervals[1::]} # The set of all indices that correspond to miRNA-encoding genes that were detected having changed gene expression significantly. Start with interval [0,x) which is indicated as key x. Then interval [x, 2*x) etc.
        not_detected = [i for i in range(self.number_of_genes)] # list of non detected genes

        # detection of out-of-steady-state genes, creating G_t (and M_t) as described in section 3.2
        for j, t in enumerate(t_intervals[0:len(t_intervals)-1]):
            to_be_removed = []
            for gene_i in not_detected:
                t_interval_time_points = np.linspace(t, t+t_interval_size, 1000)
                min_interval = min(x_fit[gene_i](t_interval_time_points))
                max_interval = max(x_fit[gene_i](t_interval_time_points))
                if (abs(min_interval - x_fit[gene_i](0)) > self.detection_threshold) or (abs(max_interval - x_fit[gene_i](0)) > self.detection_threshold):
                    to_be_removed.append(gene_i)
                    for tt in t_intervals[j+1:len(t_intervals)]:
                        G_t[tt].add(gene_i) # ADAPT HERE TO DISTINGUISH BETWEEN M_t and G_t. If gene is miRNA ecoding gene, M_t[t+1].add(i) else G_t[t+1].add(i)
                    
            for gene_j in to_be_removed:
                not_detected.remove(gene_j)

        # Plotting gene expression
        if show_plot:
            time_vs_expression(self.T, TT, x_fit, self.number_of_genes, self.rna_expression, knots, lmbda, k)

        ### P2 ###
        protein_degradation_rate = np.array([0.5 for _ in range(self.number_of_genes)]) # make all 1 (simplification) 
        protein_translation_rate = np.array([1,2,1]) # taken from experiment 4.1

        alpha_i = np.zeros((self.number_of_genes, self.D)) # weights for the B-spline basis functions to create the B-spline curve

        for g in range(self.number_of_genes):
            x_i = x_fit[g](self.T)
            alpha_i[g, :] = fit_bspline_protein_expression(protein_translation_rate[g], protein_degradation_rate[g], self.T, self.knots, self.k, self.D, x_i) # P2
        if show_plot:
            time_vs_expression(self.T, TT, x_fit, self.number_of_genes, self.rna_expression, self.knots, self.lmbda, self.k, False, alpha_i) # plotting protein expression

        ### P3 ###
        all_possible_regulators = self._powerSet(list(G_t.values())[-1])
        regulator_idx = {}
        for idx, regs in enumerate(all_possible_regulators):
            regulator_idx[regs] = idx # make sure indexing for a (pair of) regulator(s) is consistent for different sizes of G(t)

        N_t_L = len(all_possible_regulators) # largest set of regulators (out of steady-state genes) found. I.e. N(t_L) in the paper. 
        p_vector = np.zeros((self.number_of_genes, len(self.T), 2*N_t_L)) # matrix holding p vector as described in equation 11 
        for g in range(self.number_of_genes): 
            for t_idx, t in enumerate(self.T): 
                interval = self._interval_detection(t, np.array(list(G_t.keys()))) 
                protein_regulators_possibilities = self._powerSet(G_t[interval]) # point of optimalization is possible
                for regulators in protein_regulators_possibilities: 
                    y_k = 1 
                    for regulator in regulators: 
                        y_k *=  BSpline(knots, alpha_i[regulator], k=self.k)(t) 
                    p_vector[g, t_idx, regulator_idx[regulators]] = y_k 
                    if y_k < 0: 
                        print(f"t = {t}, regulators = {regulators}") 
                        print(BSpline(knots, alpha_i[regulator], k=self.k)(t))


        # Multiply with constant C(t) = lambda_i^RNA*x_i(t) + dx/dt(t)
        lmbda_RNA = [0.1, 0.1, 0.1] # taken from the paper
        delta_t = self.T[1] - self.T[0] # paper describes two ways to take the derivative.
        for g in range(self.number_of_genes):
            for t_idx, t in enumerate(self.T):
                # c_t = lmbda_RNA[g] * x_fit[g](t) + ((x_fit[g](t + delta_t) - x_fit[g](t))/delta_t)
                c_t = lmbda_RNA[g] * x_fit[g](t) + x_fit[g].derivative()(t)
                c_t_p = p_vector[g, t_idx, :N_t_L] * -c_t 
                p_vector[g, t_idx, N_t_L:] = c_t_p

        print("gene 1 [a b]")
        coef1 = P3(p_vector[0]) # outputs the a and b vector concatenated.
        print("gene 2 [a b]")
        coef2 = P3(p_vector[1])
        print("gene 3 [a b]")
        coef3 = P3(p_vector[2])

if __name__ == "__main__":

    ### Load data DREAM CHALLENGE ###
    # path_name = os.p  ath.join("data", "DREAM4", "DREAM4_InSilico_Size10", "insilico_size10_2" ,"insilico_size10_2_timeseries.tsv")
    # n1, n2, n3, n4, n5 = read_dream_time_series(path_name, 21) # returns np.arrays
    # T = n1[:,0]
    # rna_expression = n1[:, 1::]
    
    ### Load data EXPERIMENT 4.1 ###
    path_name = os.path.join("data", "experiment_4_1", "experiment_4_1.tsv")
    data = read_experiment_41(path_name)
    T = data[:,0] # first columns contains time points, rest is gene expression
    rna_expression = data[:, 1::] 

    ### try different normalization techniques ###
    rna_expression_normalized_per_gene_norm = np.empty_like(rna_expression)
    for column in range(len(rna_expression[0])):
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

    ### hyper parameters ###
    lmbda = 15
    k = 3 # degree of B-spline  
    R = 7
    detection_threshold = 0.05
    show_plot = True
    knots = np.array([0, 2, 4, 5, 10, 15, 20, 30, 50])
    knots = np.linspace(0, 50, 10)
    network = NetworkInference(T, rna_expression, lmbda, R, knots, k, detection_threshold)
    network.infer(show_plot)