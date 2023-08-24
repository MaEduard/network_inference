import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

# function that returns dz/dt
def model(z,t):
    if t <= 0:
        x1 = 0
        x2 = 0.35
        x3 = 1.65
        print('here')
    else:
        x1 = 0.3
        x2 = z[1]
        x3 = z[2]
    y1 = z[3]
    y2 = z[4]
    y3 = z[5]
    dx1dt = (0.1 + 0.05*y1*y2 + 0.025*y1*y3)/(1 + 0.1*y1 + 10*y3 + 0.05*y1*y2 + 0.025*y1*y3) - 0.1*x1
    dx2dt = (0.1 + 0.1*y1 + 0.1*y1*y2)/(1 + 0.1*y1 + 0.1*y1*y2 + 10*y1*y3) - 0.1*x2
    dx3dt = (0.1 + 0.1*y2)/(1 + 0.1*y2 + 0.1*y3) - 0.1*x3
    dy1dt = x1 - 0.5*y1
    dy2dt = 2*x2 - 0.5*y2
    dy3dt = x3 - 0.5*y3
    dzdt = [dx1dt, dx2dt, dx3dt, dy1dt, dy2dt, dy3dt]
    return dzdt

# initial condition
x0 = [0, 0.35, 1.65]
y0 = [0, 1.5, 3.4]
z0 = x0 + y0

# time points
t = np.arange(0, 50, 1)

# store solution
x1 = np.zeros(len(t))
x2 = np.zeros(len(t))
x3 = np.zeros(len(t))
y1 = np.zeros(len(t))
y2 = np.zeros(len(t))
y3 = np.zeros(len(t))

x1[:] = 0.3
# x2[0:50] = x0[1]
# x3[0:50] = x0[2]

# record initial conditions
x2[0] = x0[1]
x3[0] = x0[2]
y1[0] = y0[0]
y2[0] = y0[1]
y3[0] = y0[2]

# z = odeint(model, z0, t)
# print(z.shape)

# solve ODE
for i in range(1,len(t)):
    # span for next time step
    tspan = [t[i-1],t[i]]
    # solve for next step
    z = odeint(model,z0,tspan)
    # store solution for plotting
    x2[i] = z[1][1]
    x3[i] = z[1][2]
    y1[i] = z[1][3]
    y2[i] = z[1][4]
    y3[i] = z[1][5]
    # next initial condition
    z0 = z[1]

# x1[0] = 0

# plot results
fig, axs = plt.subplots(1, 2)
fig.suptitle('Experiment 4.1')
axs[0].plot(t,x1,'b')
axs[0].plot(t,x2,'g')
axs[0].plot(t,x3,'r')
axs[0].set_title("RNA expression")
axs[1].plot(t,y1,'b')
axs[1].plot(t,y2,'g')
axs[1].plot(t,y3,'r')
axs[1].set_title("Protein expression")

for ax in axs.flat:
    ax.set(xlabel='Time', ylabel='Expression levels')

plt.show()

all_data = np.zeros((len(x1), 4))
all_data[:,0] = t
all_data[:, 1] = x1
all_data[0, 1] = 0
all_data[:, 2] = x2
all_data[:, 3] = x3

print(all_data.shape)

sample_time_points = np.arange(0, 50, 1)
df_all_data = pd.DataFrame(all_data[sample_time_points,:], columns = ['Time','G1','G2', 'G3'])

print(df_all_data)
# df_all_data.to_csv('data/experiment_4_1/experiment_4_1.tsv', sep="\t", index=False)

# p_data = pd.DataFrame(p_data[sample_time_points,:], columns = ['Time','P1','P2', 'P3'])
# p_data = np.zeros((len(x1), 4))
# p_data[:,0] = t
# p_data[:, 1] = y1
# p_data[0, 1] = 0
# p_data[:, 2] = y2
# p_data[:, 3] = y3

# p_data = pd.DataFrame(p_data[sample_time_points,:], columns = ['Time','P1','P2', 'P3'])

# print(p_data)

# p_data.to_csv('data/experiment_4_1/experiment_4_1_protein.tsv', sep="\t", index=False)

# print(df_all_data)



# print(x1)
# print(x2)
# print(x3)