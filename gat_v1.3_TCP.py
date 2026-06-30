# libraries will be sorted later
import os
import numpy as np
import pandas as pd
import spicy
from scipy.ndimage import label
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from collections import defaultdict
from torch.optim import Adam
from sympy.codegen import numpy_nodes



# loads .mat pickle file
def get_mat(filepath):
    with open(filepath, "rb") as file:
        return pickle.load(file)


# determines threshold for removing background - probably there's better way to do this
def find_threshold(chunk, t):
    bg = np.median(chunk)
    threshold = bg + (t * np.sqrt(bg + 1))
    return threshold


def coincide(gamma1, gamma2, dict):
    return (gamma2 in dict.get(gamma1, set())
            or
            gamma1 in dict.get(gamma2, set()))


# load matrices
n = 50  # number of matrices
list_transitions = []

for k in range(n):
    matrix = get_mat(
        f'/Users/trumanparrish/PyCharmMiscProject/Isomer/data/NonBlurredArrays/yy_coincidence_matrix_clean({k}).mat')

    # break matrix into smaller pieces for faster processing and locate peaks in regions with a threshold
    mat_size = matrix.shape[0]
    chunk_size = 128
    peaks = np.zeros(matrix.shape, dtype=bool)

    for i in range(0, mat_size, chunk_size):
        for j in range(0, mat_size, chunk_size):
            chunk = matrix[i:i + chunk_size, j:j + chunk_size]
            threshold = find_threshold(chunk, 30)
            chunk_mask = chunk > threshold

            peaks[i:i + chunk_size, j:j + chunk_size] = chunk_mask

    # break the matrix into binary regions and assign a label to each region
    grouped_peaks, num_regions = label(peaks)

    # takes centroid to find actual peak of the cluster of values in each region
    actual_peaks = center_of_mass(
        matrix,  # count values
        grouped_peaks,  # regions
        range(1, num_regions + 1)  # region labels
    )

    # print peaks in .csv, make active to check peaks coords
    # np.savetxt("centroids.csv", actual_peaks, delimiter=",")

    # building adjacency list
    transitions = np.unique(actual_peaks)
    # print(transitions) #make active to check unique transitions

    # organizes dict where the key is a transition and the values with that key are the other transitions it sees
    tran_in_coinc = defaultdict(set)
    for y1, y2 in actual_peaks:
        tran_in_coinc[y1].add(y2)
        tran_in_coinc[y2].add(y1)

    list_transitions.append(tran_in_coinc)

'''
Everything before this works well and feels good
Past this is ML to build the scheme from the dict of coincidences
Current idea use transitions as nodes and the coincidences as edges and then gives those edges a feature value
Not sure what the feature value will be so far
'''
# index key values for nodes numbers
list_e_i = []
list_i_e = []
list_edges = []
list_features_node = []

for k in range(n):
    dih = list_transitions[k]
    tot_trans = set(dih.keys())

    for tran, coincs in dih.items():
        for tran2 in coincs:
            tot_trans.add(tran2)
    sorted_transitions = sorted(tot_trans)
    energy_index = {energy: ind for ind, energy in enumerate(sorted_transitions)}
    index_energy = {i: E for E, i in energy_index.items()}
    list_e_i.append(energy_index)
    list_i_e.append(index_energy)
    # create edges between nodes using the index and store in tensor

    edges = []
    for tran, coincs in dih.items():
        for tran2 in coincs:
            node1 = energy_index[tran]
            node2 = energy_index[tran2]
            edges.append((node1, node2))
    edge_index = torch.tensor(edges, dtype=torch.long).T
    list_edges.append(edge_index)

    # create node features
    node_feat = []
    for i in sorted_transitions:
        num_coincidences = len(dih.get(i, set()))
        node_feat.append([i, num_coincidences])
    #print(node_feat)
    features_node = torch.tensor(node_feat, dtype=torch.float)
    # normalize
    features_node[:, 0] = features_node[:, 0] / features_node[:, 0].max()
    features_node[:, 1] = features_node[:, 1] / features_node[:, 1].max()
    list_features_node.append(features_node)

# GAT to predict bands
class Band_Predicting_GAT(torch.nn.Module):
    def __init__(self, inputs, behind, out_dim):
        super().__init__()

        self.gat1 = GATConv(in_channels=inputs, out_channels=behind, heads=2)
        self.gat2 = GATConv(in_channels=behind * 2, out_channels=out_dim, heads=1)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)

        y = self.gat2(x, edge_index)
        return y


band_model = Band_Predicting_GAT(inputs=features_node.shape[1], behind=16, out_dim=8)

list_possibilities = []  # possible pairs
list_ratings = []  # the ratings of weather or not those pairs are coincident

for k in range(n):
    poss = []
    rat = []

    feat_node = list_features_node[k]
    fih = list_transitions[k]
    sorted_transitions = sorted(fih.keys())
    for i in range(len(sorted_transitions)):
        for j in range(i + 1, len(sorted_transitions)):
            gamma1 = sorted_transitions[i]
            gamma2 = sorted_transitions[j]

            same_pair = coincide(gamma1, gamma2, fih)

            poss.append([i, j])
            rat.append(1.0 if same_pair else 0.0)

    possibilities = torch.tensor(poss, dtype=torch.long)
    ratings = torch.tensor(rat, dtype=torch.float)
    list_possibilities.append(possibilities)
    list_ratings.append(ratings)

# optimizer is used for adjusting weights, learning rate can be adjusted lower if training is unstable
optimizer = torch.optim.Adam(band_model.parameters(), lr=0.01, weight_decay=1e-4)

loss_storage = []

for epoch in range(1000):
    band_model.train()
    total_loss = 0


    for fart in range(n):
        feat_node = list_features_node[fart]
        edge_ind = list_edges[fart]
        possibility = list_possibilities[fart]
        rats = list_ratings[fart]
        optimizer.zero_grad()
        z = band_model(feat_node, edge_ind)

        first = possibility[:, 0]
        second = possibility[:, 1]

        sim_score = (z[first] * z[second]).sum(dim=1)

        loss = F.binary_cross_entropy_with_logits(sim_score, rats)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loss_storage.append(loss.item())

    if epoch % 50 == 0:
        print("Epoch:", epoch, "Average loss:", total_loss / 50)

###
###
###
"""
x_epochs = np.linspace(0, 50, len(loss_storage))
plt.figure()
plt.plot(x_epochs, loss_storage)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Function")
plt.show()
"""

band_model.eval()

display = [1, 2,3,4,5,6] #samples to display bands for

with torch.no_grad():
    embed = band_model(features_node, edge_index)

    normalized = embed / (embed.norm(dim=1, keepdim=True))
    grouping_chances = normalized @ normalized.T

grouping_threshold = 0.75
min_size = 2

possible_bands = []

num_nodes = embed.shape[0]
for i in range(num_nodes):
    band = []

    for j in range(num_nodes):
        if grouping_chances[i, j] >= grouping_threshold:
            band.append(j)

    if len(band) >= min_size:
        possible_bands.append(band)

seen = set()
unique_groups = []

for band in possible_bands:
    key = frozenset(band)

    if key not in seen:
        seen.add(key)
        unique_groups.append(band)


for num in range(10):
    if num in display:
        index_energy = list_i_e[num]
        print("\nPossible bands for sample", num, ":")

        for band_num, band in enumerate(unique_groups):
            energies = []

            for node_idx in band:
                energy = float(index_energy[node_idx])
                energies.append(energy)

            print("Band", band_num + 1, ":", sorted(energies))