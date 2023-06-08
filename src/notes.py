
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