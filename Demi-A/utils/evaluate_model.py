import matplotlib.pyplot as plt
import pickle
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import r2_score
import numpy as np
from .GNN_data import make_undirected

def visualize_performance(y_K, pred_K, R2_K, cover_interval, overlap, save_dir):
    plt.rcParams['font.size'] = 15
    # Plotting for E
    fig, ax = plt.subplots(dpi=300, figsize=(5,5))   

    ax.plot(y_K, pred_K, 'bo', markersize=3, label=f'$R^2 = %.3f$' %R2_K)
    ax.axline((0, 0), slope=1, color='red', linestyle='--')
    ax.set_ylabel('Predicted $E$ [GPa]')
    ax.set_xlabel('Ground Truth $E$ [GPa]')


    ax.set_xlim([min(y_K), max(y_K)])
    ax.set_ylim([min(y_K), max(y_K)])
    ax.legend()

    plt.tight_layout()
    
    
def load_graph_data(data_dir, avg_scheme, batch_size, val_percentage):
    """
    function for loading graph data
    """
    with open(data_dir, 'rb') as f: 
        data_list = pickle.load(f)

    new_dataset = []
    for data in data_list:
        new_data = data.clone()
        if avg_scheme == 'Voigt':
            new_data.K = new_data.K_Voigt
            del new_data.K_Voigt
        elif avg_scheme == 'Hill':
            new_data.K = new_data.K_Hill
            del new_data.K_Hill
        new_dataset.append(new_data)
    data_list = new_dataset 
    # Convert all graphs in data_list to undirected
    data_list = [make_undirected(data) for data in data_list]
    
    for data in data_list:
        max_val = torch.round(torch.max(data.x[:,:3])).item()+1  # Compute max value for current graph
        data.x[:,:6] /= max_val                   # Normalize the first 6 columns
    
    dataset_size = len(data_list)
    valid_percentage = val_percentage
    valid_size = int(valid_percentage*dataset_size)
    valid_set, remain_set = random_split(data_list, [valid_size, dataset_size - valid_size])
    dataset = valid_set
    print(f'Number of test graphs: {len(dataset)}')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader    
    
def evaluate_model(model, loader, device, cover_interval, overlap, save_dir):
    # Evaluate the model on the test set
    model.eval()

    K_pred_whole = []
    y_whole_K = []

    with torch.no_grad():
        for data in loader:
            K_pred = model(data.x.to(device), data.edge_index.long().to(device), data.batch.to(device))
            data.K = data.K.to(torch.float32)

            K_pred_whole.append(K_pred.cpu().numpy())
            y_whole_K.append(data.K.cpu().numpy())

    K_pred_whole = np.concatenate(K_pred_whole, axis=0)
    y_whole_K = np.concatenate(y_whole_K, axis=0)

    R2_K = r2_score(y_whole_K, K_pred_whole)

    print(f'Test R2 for E: {R2_K}')

    # Return values if needed later
    visualize_performance(y_whole_K, K_pred_whole, R2_K, cover_interval, overlap, save_dir)
    return R2_K
