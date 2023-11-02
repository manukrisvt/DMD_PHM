import numpy as np
import matplotlib.pyplot as plt
from modwt import modwt, modwtmra


def plot_orginal_with_recreated(x_org, x_rec, rec_name="DMD", figsize=(15, 10)):
    fig, ax = plt.subplots(np.shape(x_org)[0], 1, figsize=figsize, sharex=True)
    if np.shape(x_org)[0] == 1:
        ax = [ax]
    # ax = ax.flatten()
    for nrow in range(0, np.shape(x_org)[0]):
        ax[nrow].plot(x_org[nrow, :])
        ax[nrow].plot(x_rec[nrow, :], linestyle="--")
        ax[nrow].set_xlabel("Snapshots")
        ax[nrow].set_ylabel("Value")
        ax[nrow].legend([f"x{nrow}-Org", f"x{nrow}-{rec_name}"])
    plt.show()


def get_sys_mat_DMD(x_states, y=None, u_inp=None, l2=10e-12, plotting=True):
    if (y is None) and (u_inp is None):  # Vanilla DMD
        u, d, vh = np.linalg.svd(x_states[:, :-1], full_matrices=False)
        v = np.transpose(vh)
        if plotting:
            plt.figure(figsize=(10, 6))
            #     print(d)
            plt.plot(d / d[0], "xr")
            plt.yscale("log")
            plt.axhline(y=l2)
            plt.xlabel("Singular values")
            plt.ylabel("Decay of SV")
            plt.tight_layout()
        s = len(np.where(d / d[0] > l2)[0])
        print(s)
        n = np.shape(x_states)[0]
        print(n)
        u = u[:, :s]
        v = v[:, :s]
        di_arr = 1 / d
        #   print(di_arr)
        di = np.diag(di_arr[:s])
        A = np.linalg.multi_dot([x_states[:, 1:], v, di, np.transpose(u[0:n, :])])
        return A
    elif (y is not None) and (u_inp is None):  # Output DMD (Input not implemented)
        u, d, vh = np.linalg.svd(x_states[:, :-1], full_matrices=False)
        v = np.transpose(vh)
        if plotting:
            plt.figure(figsize=(10, 6))
            plt.plot(d / d[0], "xr")
            plt.axhline(y=l2)
            plt.yscale("log")
            plt.xlabel("Singular values")
            plt.ylabel("Decay of SV")
            plt.tight_layout()
        s = len(np.where(d / d[0] > l2)[0])
        print(s)
        n = np.shape(x_states)[0]
        print(n)
        u = u[:, :s]
        v = v[:, :s]
        di_arr = 1 / d
        #   print(di_arr)
        di = np.diag(di_arr[:s])
        A = np.linalg.multi_dot([x_states[:, 1:], v, di, np.transpose(u[0:n, :])])
        # B = np.linalg.multi_dot([X2,v,di,np.transpose(u[n:,:])])
        C = np.linalg.multi_dot([y[:, :-1], v, di, np.transpose(u[0:n, :])])
        # D = np.linalg.multi_dot([y1,v,di,np.transpose(u[n:,:])])
        return A, C
    elif (y is not None) and (u_inp is not None):
        x1 = x_states[:, :-1]
        u1 = u_inp[:, :-1]

        x1u1 = np.vstack([x1, u1])
        u, d, vh = np.linalg.svd(x1u1, full_matrices=False)
        v = np.transpose(vh)
        if plotting:
            plt.figure(figsize=(10, 6))
            plt.plot(d / d[0], "xr")
            plt.axhline(y=l2)
            plt.yscale("log")
            plt.xlabel("Singular values")
            plt.ylabel("Decay of SV")
            plt.tight_layout()
        s = len(np.where(d / d[0] > l2)[0])
        print(s)
        n = np.shape(x_states)[0]
        u = u[:, :s]
        v = v[:, :s]
        di_arr = 1 / d
        #   print(di_arr)
        di = np.diag(di_arr[:s])
        A = np.linalg.multi_dot([x_states[:, 1:], v, di, np.transpose(u[0:n, :])])
        B = np.linalg.multi_dot([x_states[:, 1:], v, di, np.transpose(u[n:, :])])
        C = np.linalg.multi_dot([y[:, :-1], v, di, np.transpose(u[0:n, :])])
        D = np.linalg.multi_dot([y[:, :-1], v, di, np.transpose(u[n:, :])])
        return A, B, C, D
    else:
        print("Not implemented yet")


# End of file