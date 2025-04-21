import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, random_split
from itertools import permutations
import networkx as nx
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(data_dir):
    with open(data_dir, 'rb') as f: 
        return pickle.load(f)
    
def Elastic_property_avg_scheme(data_list, scheme):
    new_dataset = []
    for data in data_list:
        new_data = data.clone()
        if scheme == 'Voigt':
            new_data.K = new_data.K_Voigt
            new_data.mu = new_data.mu_Voigt
            del new_data.K_Voigt, new_data.mu_Voigt
        elif scheme == 'Hill':
            new_data.K = new_data.K_Hill
            new_data.mu = new_data.mu_Hill
            del new_data.K_Hill, new_data.mu_Hill
        new_dataset.append(new_data)
    return new_dataset 

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

def feature_engineering(data_list, cluster_data=None):

    engineered_data_list = []
    for data in data_list:
        new_data = data.clone()

        # --- Center Coordinates (x, y) ---
        center_coords = data.x[:, :2]

        # --- Node Degree ---
        G = nx.Graph()
        G.add_edges_from(new_data.edge_index.t().tolist())
        degrees = torch.tensor([G.degree[i] for i in range(new_data.num_nodes)], dtype=torch.float).view(-1, 1)

        # --- Closeness Centrality ---
        closeness = torch.tensor(list(nx.closeness_centrality(G).values()), dtype=torch.float).view(-1, 1)

        # --- Eigenvector Centrality ---
        eigenvector = torch.tensor(list(nx.eigenvector_centrality(G).values()), dtype=torch.float).view(-1, 1)

        # --- PageRank ---
        pagerank = torch.tensor(list(nx.pagerank(G).values()), dtype=torch.float).view(-1, 1)

        # --- Combine Features ---
        new_data.x = torch.cat([center_coords, degrees, closeness, eigenvector, pagerank], dim=1)

        engineered_data_list.append(new_data)

    return engineered_data_list

def normalize_planar_info(data_list):
    for data in data_list:
        max_val = torch.round(torch.max(data.x[:,:2])).item()+1  # Compute max value for current graph
        data.x[:,:2] /= max_val                   # Normalize the first 4 columns
    return data_list

def normalize_k(data_list):
    """
    Log-transforms and then normalizes the K values in the data_list using max scaling.
    """
    # Log-transform K values (ensure no zero or negative values)
    K_values = np.array([np.log(data.K) for data in data_list])  # Adding small value to avoid log(0)
    
    # Max scaling: Divide by max value
    K_max = np.max(K_values)
    K_scaled = K_values / K_max

    # Assign back to data objects
    for i, data in enumerate(data_list):
        data.K = K_scaled[i]

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
