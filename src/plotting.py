import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

def time_vs_expression(T, TT, x_fit, number_of_genes, rna_expression, knots, lmbda, k, gene_expression=True, alpha_i=[]):
    # plotting settings
    # colors = ['b', 'r', 'lime', 'magenta', 'black', 'sienna', 'darkolivegreen', 'turquoise', 'hotpink', 'goldenrod']
    colors = ['b', 'lime', 'r']
    gene_numbers = [i for i in range(number_of_genes)]
    
    # create plotting points
    for i in range(number_of_genes):
        prot_expr = []
        # if you want to plot rna expression, else protein expression is plotted.
        if gene_expression:
            y_pred = x_fit[gene_numbers[i]](TT)
            plt.plot(T, rna_expression[:,gene_numbers[i]], label=f"gene {gene_numbers[i]}", color=colors[i % len(colors)], linewidth=1.0)
            plt.plot(TT, y_pred, '--', color=colors[i % len(colors)], label=f"fitted gene {gene_numbers[i]}", linewidth=1.0)
        else:
            for t in TT:
                prot_expr.append(BSpline(knots, alpha_i[i], k)(t))
            plt.plot(TT, prot_expr, label=f"gene {i}", color=colors[i % len(colors)], linewidth=1.0)
    
    # plotting
    if gene_expression:
        plt.title(f"Gene expression for $\lambda$ = {lmbda}, $k$ = {k}, knots = {len(knots)}")
    else:
        plt.title(f"Protein expression for $\lambda$ = {lmbda}, $k$ = {k}, knots = {len(knots)}")
    plt.ylabel("Expresssion levels")
    plt.xlabel("Time")
    plt.legend(loc='upper right', fancybox = True, shadow=True)
    plt.grid()
    plt.show()