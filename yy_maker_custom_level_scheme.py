import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import scipy
import os
from pathlib import Path
from level_scheme_plotter import plot_branching_level_scheme


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


def apply_compton(gamma_, chance_to_scatter=0.9):
    gamma = gamma_
    cross_gamma = 0
    if np.random.rand() < chance_to_scatter:
        # Compton Escape Scattering
        gamma = gamma * np.random.rand()

    if np.random.rand() < chance_to_scatter * 1 / 5:
        # Compton Cross Scattering
        p = np.random.rand()
        x = lerp(0, gamma, p)
        cross_gamma = gamma - x
        gamma = x

        if np.random.rand() < chance_to_scatter:
            # Compton Escape Scattering when entering the second detector
            gamma = gamma * np.random.rand()

    return (int(gamma), int(cross_gamma))



def CreateNewMat(num_gamma_rays, size_mat, level_scheme, chance_to_scatter):
    #Create a matrix with and without scattering for peak comparison
    coinc_mat = np.zeros((size_mat, size_mat), dtype=np.float32)
    coinc_mat_no_scatter = np.zeros((size_mat, size_mat), dtype=np.float32)
 
    #Create the actual decay sequence
    #FIX 1: break before appending so the ground state energy (0) is never added
    n_levels = len(level_scheme)
    sequence = {}
    for start in range(1, n_levels):
        gammas = []
        current_idx = int(start)
 
        while current_idx != -1:
            gamma_e, next_idx, _ = level_scheme[current_idx]
            next_idx = int(next_idx)
 
            if next_idx == -1:
                break
 
            gamma_e = int(gamma_e)
 
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
 
        #Apply Compton scattering to each gamma individually (not per-pair)
        scattered_gammas = []
        for gamma in gammas:
            scattered_e, _ = apply_compton(gamma, chance_to_scatter)
            scattered_gammas.append(scattered_e)
 
        #Start filling the matrices
        for i in range(len(gammas)):
            for j in range(i + 1, len(gammas)):
                #Always adds to non-scattered matrix (energies already bounds-checked above)
                x, y = gammas[i], gammas[j]
                coinc_mat_no_scatter[x][y] += 1
                coinc_mat_no_scatter[y][x] += 1
 
                #FIX 2: strict lower bound discards gammas scattered below detection threshold
                x, y = scattered_gammas[i], scattered_gammas[j]
                if 0 < x < size_mat and 0 < y < size_mat:
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
    # ev_levels are actual gamma energies, not the energies themselves
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
        # Added coinc_mat_no_scatter (CreateNewMat now has 2 outputs)
        coinc_mat, coinc_mat_no_scatter = CreateNewMat(num_gamma_rays, arr_size, transition_scheme.copy(), 0)

    transition_mat = CreateTransitionMatFromTransitionScheme(transition_scheme.copy())

    # Also return the transition scheme to create the level schemes later

    return coinc_mat, coinc_mat_no_scatter, transition_mat, transition_scheme

def _add_compton_ridge(coinc_mat, continuum_gamma, fixed_gamma, size_mat):
    """
    Distribute 1 count uniformly along the Compton continuum.
    continuum_gamma : the gamma that scattered (energy spans 1 → original)
    fixed_gamma     : the gamma fully detected (fixes the row/column)
    """
    if fixed_gamma <= 0 or fixed_gamma >= size_mat or continuum_gamma <= 1:
        return
    weight = 1.0 / continuum_gamma
    for e in range(1, min(continuum_gamma, size_mat)):
        coinc_mat[e][fixed_gamma] += weight
        coinc_mat[fixed_gamma][e] += weight


def main(create_new_mat=True, sim_size=0, custom_transition_scheme=True, out_dir="./data", n_matrices=1000):
    gbs = [(5, 0.5)]
    transition_schemes = []

    if n_matrices == 1:
        out_dir = "."
        coinc_mat, coinc_mat_clean, transition_mat, transition_scheme = CreateSample(3, True)

        plot_branching_level_scheme(transition_scheme, "Level Scheme")

        SaveMat(coinc_mat, os.path.join(out_dir, "yy_coincidence_matrix.mat"))
        SaveMat(coinc_mat_clean, os.path.join(out_dir, "yy_coincidence_matrix_clean.mat"))
        SaveMat(transition_mat, os.path.join(out_dir, "transition_matrix.mat"))
    
        #Generate matrices with no blur
        coinc_mat_no_blur = np.clip(np.copy(coinc_mat), 1, None)
        coinc_mat_clean_no_blur = np.clip(np.copy(coinc_mat_clean), 1, None)

        SaveGraph(
            coinc_mat_no_blur,
            out_dir,
            "Sample Generated YY Matrix",
            norm="log",
            vmin=None,
            vmax=None,
        )
        SaveGraph(
            coinc_mat_clean_no_blur,
            out_dir,
            "Sample Generated YY Matrix No Scatter",
            norm="log",
            vmin=None,
            vmax=None,
        )

        for gblur in gbs:
            gb_size = gblur[0]
            gb = np.zeros((gb_size, gb_size))

            for gb_i in range(gb_size):
                for gb_j in range(gb_size):
                    gb[gb_i][gb_j] = Gauss2D(
                        2 * (gb_i - gb_size / 2) / gb_size,
                        0,
                        gblur[1],
                        2 * (gb_j - gb_size / 2) / gb_size,
                        0,
                        gblur[1],
                    )

            gb = gb / gb.sum()

            coinc_mat = convolve(coinc_mat, gb)
            coinc_mat_clean = convolve(coinc_mat_clean, gb)

            blurry_coinc_mat = np.copy(coinc_mat)
            coinc_mat_copy = np.clip(np.copy(blurry_coinc_mat), 1, None)
            blurry_coinc_mat_clean = np.copy(coinc_mat_clean)
            coinc_mat_clean_copy = np.clip(np.copy(blurry_coinc_mat_clean), 1, None)

            SaveGraph(
                coinc_mat_copy,
                out_dir,
                f"Sample Generated YY Matrix (gb_size = {gb_size}, std = {gblur[1]})",
                norm="log",
                vmin=None,
                vmax=None,
                )
            SaveGraph(
                coinc_mat_clean_copy,
                out_dir,
                f"Sample Generated YY Matrix No Scatter (gb_size = {gb_size}, std = {gblur[1]})",
                norm="log",
                vmin=None,
                vmax=None,
                )
            
            SaveMat(coinc_mat, os.path.join(out_dir, f"gaussian_coincidence_matrix({gb_size}, {gblur[1]}).mat"))
            SaveMat(coinc_mat_clean, os.path.join(out_dir, f"gaussian_coincidence_matrix_clean({gb_size}, {gblur[1]}).mat"))

        return transition_schemes


    for i in range(n_matrices):
        coinc_mat, coinc_mat_clean, transition_mat, transition_scheme = CreateSample(3, True)

        plot_branching_level_scheme(transition_scheme, f"Level Scheme ({i})")

        transition_schemes.append(transition_scheme)

        SaveMat(coinc_mat, os.path.join(f"{out_dir}/NonBlurredArrays", f"yy_coincidence_matrix({i}).mat"))
        SaveMat(coinc_mat_clean, os.path.join(f"{out_dir}/NonBlurredArrays", f"yy_coincidence_matrix_clean({i}).mat"))
        SaveMat(transition_mat, os.path.join(f"{out_dir}/TransitionMats", f"transition_matrix({i}).mat"))
    
        #Generate matrices with no blur
        coinc_mat_no_blur = np.clip(np.copy(coinc_mat), 1, None)
        coinc_mat_clean_no_blur = np.clip(np.copy(coinc_mat_clean), 1, None)

        SaveGraph(
            coinc_mat_no_blur,
            f"{out_dir}/NoBlurGraphs",
            f"Sample Generated YY Matrix ({i})",
            norm="log",
            vmin=None,
            vmax=None,
        )
        SaveGraph(
            coinc_mat_clean_no_blur,
            f"{out_dir}/NoBlurGraphs",
            f"Sample Generated YY Matrix No Scatter ({i})",
            norm="log",
            vmin=None,
            vmax=None,
        )

    return transition_schemes
       
if __name__ == "__main__":
    transition_schemes = main(n_matrices=50, out_dir="./Isomer-ARL/toy_models/test_matrices")