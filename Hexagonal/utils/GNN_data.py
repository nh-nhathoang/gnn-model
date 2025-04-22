import torch
import pickle
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, random_split
from itertools import permutations
import networkx as nx
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

    
def is_undirected(data):
    # Extract the edge index
    edge_index = data.edge_index
    
    # Create a set to store unique edges
    edge_set = set()
    for i in range(edge_index.shape[1]):
        # Get nodes from edge index
        node_a = edge_index[0, i].item()
        node_b = edge_index[1, i].item()

        # Check if the reverse edge exists
        if (node_b, node_a) not in edge_set:
            edge_set.add((node_a, node_b))
        else:
            edge_set.remove((node_b, node_a))
            
    # If all edges had their reverse, the edge_set should be empty
    return len(edge_set) == 0

def make_undirected(data):
    # Extract edge index
    edge_index = data.edge_index

    # Convert edge_index to a set of tuples for O(1) look-up time
    edge_set = {(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.shape[1])}

    # Identify missing reverse edges
    missing_edges = [(b, a) for a, b in edge_set if (b, a) not in edge_set]

    # Convert lists of missing edges to tensor format and add them to the original edge_index
    missing_edges_tensor = torch.tensor(missing_edges, dtype=torch.long).t()
    data.edge_index = torch.cat([edge_index, missing_edges_tensor], dim=1)
    
    return data

def feature_engineering(data_list):
    engineered_data_list = []

    for data in data_list:
        new_data = data.clone()

        # --- Center Coordinates (x, y) ---
        center_coords = new_data.x[:, :2]
        centroid = torch.mean(center_coords, dim=0, keepdim=True)
        dist_to_center = torch.norm(center_coords - centroid, dim=1, keepdim=True)

        # --- Graph Construction ---
        G = nx.Graph()
        G.add_edges_from(new_data.edge_index.t().cpu().numpy())

        # --- Node Degree ---
        degrees = torch.tensor(
            [G.degree[i] for i in range(new_data.num_nodes)], dtype=torch.float
        ).view(-1, 1)

        # --- Closeness Centrality ---
        closeness = torch.tensor(
            list(nx.closeness_centrality(G).values()), dtype=torch.float
        ).view(-1, 1)

        # --- Eigenvector Centrality ---
        eigenvector = torch.tensor(
            list(nx.eigenvector_centrality(G).values()), dtype=torch.float
        ).view(-1, 1)

        # --- PageRank ---
        pagerank = torch.tensor(
            list(nx.pagerank(G).values()), dtype=torch.float
        ).view(-1, 1)

        # --- Clustering Coefficient ---
        clustering = torch.tensor(
            list(nx.clustering(G).values()), dtype=torch.float
        ).view(-1, 1)

        new_data.x = torch.cat([center_coords, dist_to_center, degrees, 
                                closeness, clustering], dim=1)

        engineered_data_list.append(new_data)

    return engineered_data_list

def normalize_planar_info(data_list):
    for data in data_list:
        num_features = data.x.shape[1]
        for i in range(0, num_features):
            max_val = torch.max(data.x[:, i]).item()
            if max_val != 0:
                data.x[:, i] /= max_val

    return data_list


def normalize_k(data_list):

    # Assume `data_list` is your list of graph data objects
    K_vals = np.array([data.K for data in data_list]).reshape(-1, 1)  # shape: (n, 1)

    # Fit scaler
    scaler = StandardScaler()
    K_scaled = scaler.fit_transform(K_vals)

    # Assign scaled K back (as float, not tensor)
    for i, data in enumerate(data_list):
        data.K = float(K_scaled[i])

    # Save the scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    return data_list

def prepare_dataset(data_list, batch_size, train_percentage=0.80, test_percentage=0.1):
    dataset_size = len(data_list)
    train_size = int(train_percentage * dataset_size)
    test_size = int(test_percentage * dataset_size)
    valid_size = dataset_size - train_size - test_size

    train_set, test_set, valid_set = random_split(data_list, [train_size, test_size, valid_size])
    
    print(f'Number of training graphs: {len(train_set)}')
    print(f'Number of test graphs: {len(test_set)}')
    print(f'Number of vali graphs: {len(valid_set)}')
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, valid_loader
