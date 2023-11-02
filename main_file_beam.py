#%%
import pandas as pd
from modwt import modwt, modwtmra
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from support_functions import get_sys_mat_DMD, create_pseudo_wavelet_states, plot_orginal_with_recreated

#%%
data = sio.loadmat("./impulse_response_beam.mat")
x = np.transpose(data["ximpulse"])
print(f"shape of input data {np.shape(x)}")

# %%
sns.set_theme("poster")
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(np.transpose(x[::20, :]))
ax.set_xlabel("Snapshots")
ax.set_ylabel("Value")
# ax.set_xlim([0, 100])
ax.legend(["x0", "x1", "x2"])
plt.show()



# %% DMD
## Generate system matrix from the data

ss_a_dmd = get_sys_mat_DMD(x)
# Recreate the data using the learned system matrix A
x0 = x[:, 0]
rec_x = np.zeros(np.shape(x))
rec_x[:, 0] = x0
for ni in range(1, np.shape(x)[1]):
    rec_x[:, ni] = np.linalg.multi_dot([ss_a_dmd, rec_x[:, ni - 1]])

plot_orginal_with_recreated(x[::20, :], rec_x[::20, :], figsize=(20, 40))


# %% WDMD
## Mode shapes of the beam

dt = data["timpulse"][1] - data["timpulse"][0]
[eig_val, eig_vect] = np.linalg.eig(ss_a_dmd)
Vpca1 = (np.imag(np.log((eig_val)))) / (dt) / (2 * np.pi)
sorted_nat_freq = np.sort(Vpca1)
sorted_nat_freq_idx = np.argsort(Vpca1)
DMD_modes_amp = eig_val[sorted_nat_freq_idx]
DMD_modes = eig_vect[:, sorted_nat_freq_idx]

## Obtain greater than zero frequency
greater_than_zero_idx = sorted_nat_freq > 0
suspected_freq = sorted_nat_freq[greater_than_zero_idx]
suspected_modes = DMD_modes[:, greater_than_zero_idx]
suspected_modes_amp = DMD_modes_amp[greater_than_zero_idx]


physical_modes_amp_idx = (1 - np.abs(suspected_modes_amp)) < 0.01
physical_modes_amp = suspected_modes_amp[physical_modes_amp_idx]
physical_freq = suspected_freq[physical_modes_amp_idx]
physical_modes = np.imag(suspected_modes[:, physical_modes_amp_idx])


# %% Plot the physical mode shapes obtained from the data
n_mode_shape = 5
fig, ax = plt.subplots(figsize=(20, 20))
legend_label = []
for ni in range(0, n_mode_shape):
    ax.plot(
        physical_modes[np.arange(0, 60, 2), ni]
        / np.max(np.abs(physical_modes[np.arange(0, 60, 2), ni]))
    )
    legend_label.append(f"Mode-{ni+1}")
ax.legend(legend_label)

# %% IODMD with controls

#%%


chirp_data = sio.loadmat("./sine_response_v2.mat")
x = np.transpose(chirp_data["X2"])
print(f"shape of input data {np.shape(x)}")
u = chirp_data["u2"]
y = np.transpose(chirp_data["y2"])
t = np.transpose(chirp_data["t2"])

fig, ax = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
ax[0].plot(np.transpose(t), np.transpose(y))
ax[0].set_xlabel("time (t)")
ax[0].set_ylabel("Displacement")

ax[1].plot(np.transpose(t), np.transpose(u))
ax[1].set_xlabel("time (t)")
ax[1].set_ylabel("Input chirp")

plt.show()

# %%
ss_a_iodmd, ss_b_iodmd, ss_c_iodmd, ss_d_iodmd = get_sys_mat_DMD(x, y=y, u_inp=u, l2=10e-8)

# %%
# Recreate the data using the learned system matrix A
x0 = x[:, 0]
rec_x = np.zeros(np.shape(x))
rec_x[:, 0] = x0
rec_y = np.zeros(np.shape(y))
for ni in range(0, np.shape(x)[1] - 1):
    rec_x[:, ni + 1] = np.linalg.multi_dot([ss_a_iodmd, rec_x[:, ni]]) + np.linalg.multi_dot([ss_b_iodmd, u[:, ni]])
    rec_y[:, ni] = np.linalg.multi_dot([ss_c_iodmd, rec_x[:, ni + 1]]) + np.linalg.multi_dot([ss_d_iodmd, u[:, ni]])

# %%
plot_orginal_with_recreated(y, rec_y, figsize=(20, 10))

