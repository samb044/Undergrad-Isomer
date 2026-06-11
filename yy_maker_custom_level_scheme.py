import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import scipy
from scipy.ndimage import gaussian_filter
import os


def lerp(a, b, p):
    return b * p + a * (1 - p)


def SaveMat(data, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(data, file)

    print(f"Saved Mat {file_name}")


def LoadMat(file_name):
    with open(file_name, "rb") as file:
        data = pickle.load(file)
    return np.copy(data)


def Gauss(x, mu, sigma):
    return (
        1
        / np.sqrt(2 * np.pi * sigma * sigma)
        * np.exp(-((x - mu) ** 2) / (2 * sigma * sigma))
    )


def Gauss2D(x, xmu, xsigma, y, ymu, ysigma):
    return Gauss(x, xmu, xsigma) * Gauss(y, ymu, ysigma)


def SaveGraph(
    input, out_dir=".", fig_name="Matrix", cmap="hot", norm="linear", vmin=0, vmax=1
):
    fig, ax = plt.subplots(figsize=(10, 10))

    if norm == "linear":
        counts = ax.imshow(input, cmap=cmap, norm=norm)
    elif norm == "log":
        counts = ax.imshow(input, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    ax.yaxis.set_inverted(False)

    fig.suptitle(fig_name)
    fig.colorbar(counts, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{fig_name}.svg"))
    plt.close()
    print(f"Saved Graph {fig_name}")


def ProjectionGraph(input, fig_name="Matrix"):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.yaxis.set_inverted(False)

    projectionx = input.max(axis=0)
    projectiony = input.max(axis=1)

    ax.plot(
        np.linspace(0, len(projectionx), len(projectionx)),
        projectionx,
        label="Projection X",
    )
    ax.plot(
        np.linspace(0, len(projectiony), len(projectiony)),
        projectiony,
        label="Projection Y",
    )

    ax.legend()
    fig.suptitle(fig_name)
    fig.tight_layout()
    fig.savefig(f"{fig_name}.png")
    plt.close()
    print(f"Saved Graph {fig_name}")


def LineGraph(input, fig_name="Matrix"):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.yaxis.set_inverted(False)
    ax.plot(np.linspace(0, len(input), len(input)), input, label="Projection")

    ax.legend()
    fig.suptitle(fig_name)
    fig.tight_layout()
    fig.savefig(f"{fig_name}.png")
    plt.close()
    print(f"Saved Graph {fig_name}")


def convolve(input, kernel):
    output = scipy.signal.convolve(input, kernel, mode="same")
    return output


def apply_compton(x, y, chance_to_scatter=0.9):
    if np.random.rand() < chance_to_scatter:
        # Compton Escape Scattering
        if np.random.rand() < 0.5:
            x = x * np.random.rand()
            y = y
        else:
            x = x
            y = y * np.random.rand()
    if np.random.rand() < chance_to_scatter * 1 / 3:
        # Compton Cross Scattering
        p = np.random.rand()
        xo = x
        yo = y
        x = lerp(xo, yo, p)
        y = lerp(xo, yo, (1 - p))

        if np.random.rand() < chance_to_scatter:
            # Compton Escape Scattering when entering the second detector
            if np.random.rand() < 0.5:
                x = x * np.random.rand()
                y = y
            else:
                x = x
                y = y * np.random.rand()
    return (int(x), int(y))


def CreateNewMat(num_gamma_rays, size_mat, level_scheme, chance_to_scatter):
    #Create a matrix with and without scattering for peak comparison
    coinc_mat = np.zeros((size_mat, size_mat))
    coinc_mat_no_scatter = np.zeros((size_mat, size_mat))

    #Create the actual decay sequence
    n_levels = len(level_scheme)
    sequence = {}
    for start in range(1, n_levels):
        gammas = []
        current_idx = int(start)
 
        while current_idx != -1:
            ev_current, next_idx, _ = level_scheme[current_idx]
            next_idx = int(next_idx)
 
            if next_idx == -1:
                break
 
            ev_next, _, _ = level_scheme[next_idx]
            gamma_e = int(ev_current) - int(ev_next)

            if 0 < gamma_e < size_mat:
                gammas.append(gamma_e)
 
            current_idx = next_idx
 
        sequence[start] = gammas

    #Fills both scattered and non-scattered matrices
    for _ in range(num_gamma_rays):
        #Start at a random energy level and its list of emissions
        start_idx = np.random.randint(1, n_levels)
        gammas = sequence[start_idx]

        if len(gammas) < 2:
            continue

        #Start filling the matrices
        for i in range(len(gammas)):
            for j in range(i + 1, len(gammas)):
                x, y = gammas[i], gammas[j]

                #Always adds to non-scattered matrix
                if 0 <= x < size_mat and 0 <= y < size_mat:
                    coinc_mat_no_scatter[x][y] += 1
                    coinc_mat_no_scatter[y][x] += 1

                #Apply the comptom scattering and add those scattered values to the coincidence matrix
                if chance_to_scatter > 0:
                    x, y = apply_compton(x, y, chance_to_scatter)

                if 0 <= x < size_mat and 0 <= y < size_mat:
                    coinc_mat[x][y] += 1
                    coinc_mat[y][x] += 1

    print("Done Generating Matrices")

    #Return both a scattered and non-scattered matrix for comparison
    return np.copy(coinc_mat), np.copy(coinc_mat_no_scatter)


def CreateCustomTransitionScheme(specify_by_index=True):
    transition_scheme = []

    print("Not Implemented")
    gamma, next_gamma = [], []

    for i, transition in enumerate(gamma):
        next_transition = []
        for j in next_gamma[i]:
            if j in gamma:
                next_transition.append(gamma.index(j))
            else:
                next_transition.append(-1)

        transition_scheme.append((transition, next_transition, i))

    return gamma.copy(), transition_scheme.copy()


def CreateNewTransitionScheme(ev_levels):
    transition_scheme = []

    for i, level in enumerate(ev_levels):
        if i > 4 and np.random.rand() < 0.30:
            next_level = i - np.random.randint(2, i - 2)
            if next_level <= 0:
                next_level = 0
        else:
            next_level = i - 1
        transition_scheme.append((level, next_level, i))

    transition_scheme[0] = (transition_scheme[0][0], -1, transition_scheme[0][2])
    return np.copy(transition_scheme)


def CreateTransitionMatFromTransitionScheme(transition_scheme):
    transition_mat = np.zeros((len(transition_scheme), len(transition_scheme)))

    for i, level in enumerate(transition_scheme):
        ev_level, transition_idx, idx = level
        if transition_idx == -1:
            continue

        transition_mat[i][transition_idx] = 1

    return np.copy(transition_mat)


def CreateSample(sim_size, draw_background, custom_transition_scheme=False):
    num_levels = 1
    if sim_size == 0:
        arr_size = 2048
        num_levels = 12
        num_gamma_rays = 25_000
    elif sim_size == 1:
        arr_size = 1024
        num_levels = 12
        num_gamma_rays = 1_250_000
    elif sim_size == 2:
        arr_size = 2048
        num_levels = 25
        num_gamma_rays = 2_500_000
    else:
        arr_size = 4096
        num_levels = 50
        num_gamma_rays = 2_000_000

    if custom_transition_scheme:
        ev_levels, transition_scheme = CreateCustomTransitionScheme(True)
    else:
        ev_levels = []
        ev_levels.append(0)
        for i in range(num_levels):
            eV = min(int(np.random.rayleigh(arr_size / 3 + 20)), arr_size - 20)
            while eV in ev_levels:
                eV = min(
                    int(np.random.rayleigh(arr_size / 3 + 20)),
                    arr_size - 20,
                )

            ev_levels.append(eV)

        transition_scheme = CreateNewTransitionScheme(np.copy(ev_levels))

    if draw_background:
        coinc_mat, coinc_mat_no_scatter = CreateNewMat(
            num_gamma_rays, arr_size, transition_scheme.copy(), 0.45
        )
    else:
        coinc_mat, coinc_mat_no_scatter = CreateNewMat(num_gamma_rays, arr_size, transition_scheme.copy(), 0)

    transition_mat = CreateTransitionMatFromTransitionScheme(transition_scheme.copy())

    return coinc_mat, coinc_mat_no_scatter, transition_mat


def main(create_new_mat=True, sim_size=0, custom_transition_scheme=True, out_dir="."):
    coinc_mat, coinc_mat_no_scatter, transition_mat = CreateSample(3, True)

    SaveMat(coinc_mat, os.path.join(out_dir, "yy_coincidence_matrix.mat"))
    SaveMat(coinc_mat_no_scatter, os.path.join(out_dir, "yy_coincidence_matrix_no_scatter.mat"))
    SaveMat(transition_mat, os.path.join(out_dir, "transition_matrix.mat"))
    gb_size = 5
    gb = np.zeros((gb_size, gb_size))

    for gb_i in range(gb_size):
        for gb_j in range(gb_size):
            gb[gb_i][gb_j] = Gauss2D(
                2 * (gb_i - gb_size / 2) / gb_size,
                0,
                0.5,
                2 * (gb_j - gb_size / 2) / gb_size,
                0,
                0.5,
            )

    coinc_mat = convolve(coinc_mat, gb)
    coinc_mat_no_scatter = convolve(coinc_mat_no_scatter, gb)

    ep = 10**-6
    blurry_coinc_mat = np.copy(coinc_mat)
    coinc_mat_copy = np.clip(np.copy(blurry_coinc_mat), 1, None)
    coinc_mat_gaussian = gaussian_filter(coinc_mat_copy, sigma=2)

    blurry_coinc_mat_no_scatter = np.copy(coinc_mat_no_scatter)
    coinc_mat_no_scatter_copy = np.clip(np.copy(blurry_coinc_mat_no_scatter), 1, None)
    coinc_mat_gaussian_no_scatter = gaussian_filter(coinc_mat_no_scatter_copy, sigma=2)

    coinc_mat_gaussian_power = np.power(coinc_mat_gaussian, 1.5)
    
    SaveGraph(
        coinc_mat_copy,
        out_dir,
        "Sample Generated YY Matrix",
        norm="log",
        vmin=None,
        vmax=None,
        )
    SaveGraph(
        coinc_mat_gaussian,
        out_dir,
        "Sample Generated YY Matrix (Gaussian Filter)",
        norm="log",
        vmin=None,
        vmax=None,
        )

    SaveGraph(
        coinc_mat_no_scatter_copy,
        out_dir,
        "Sample Generated YY Matrix (No Scattering)",
        norm="log",
        vmin=None,
        vmax=None,
    )
    SaveGraph(
        coinc_mat_gaussian_no_scatter,
        out_dir,
        "Sample Generated YY Matrix (No Scattering Gaussian)",
        norm="log",
        vmin=None,
        vmax=None,
    )

    SaveGraph(
        coinc_mat_gaussian_power,
        out_dir,
        "Sample Generated YY Matrix (Scattering Power)",
        norm="log",
        vmin=None,
        vmax=None,
    )

    SaveMat(coinc_mat, os.path.join(out_dir, "gaussian_coincidence_matrix.mat"))


if __name__ == "__main__":
    main()
