import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device='cuda', num_epochs=1):
    model.to(device)

    train_losses = []
    test_losses = []
    train_acc_total = [] 
    test_acc_total = []
    R2_trainings = []
    R2_tests = []
    best_state_dict = None
    best_loss = float('inf')
    
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        total_graphs = 0
        pred_whole = []
        K_whole = []
        train_acc = []
        
        for data in train_loader:
            pred = model(data.x.to(device), data.edge_index.long().to(device), data.batch.to(device))
            data.K = data.K.to(torch.float32)
            
            loss = criterion(pred, data.K.to(device))
            acc = 100 - torch.abs((data.K.to(device) - pred) / data.K.to(device)) * 100
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # Gradient clipping
            optimizer.step()
            
            num_graphs_in_batch = torch.unique(data.batch).size(0)
            train_loss += loss.item() * num_graphs_in_batch
            total_graphs += num_graphs_in_batch
            
            pred_whole.append(pred.detach().cpu().numpy())
            K_whole.append(data.K.cpu().numpy())
            train_acc.append(acc.detach().cpu().numpy())
        
        out_whole = np.concatenate(pred_whole, axis=0)
        K_whole = np.concatenate(K_whole, axis=0)  
        R2_train = r2_score(K_whole, out_whole)  
        R2_trainings.append(R2_train)  
        train_acc_total.append(np.mean(np.concatenate(train_acc, axis=0)))
        
        train_losses.append(train_loss / total_graphs)
        epoch_loss = train_loss / total_graphs
        
        scheduler.step(torch.tensor(epoch_loss).float())
        
        model.eval()
        with torch.no_grad():
            pred_whole = []
            K_whole = []
            test_acc = []
            test_loss = 0
            total_graphs = 0
            
            for data in test_loader:
                pred = model(data.x.to(device), data.edge_index.long().to(device), data.batch.to(device))
                data.K = data.K.to(torch.float32)
                
                num_graphs_in_batch = torch.unique(data.batch).size(0)
                test_loss += criterion(pred, data.K.to(device)).item() * num_graphs_in_batch
                total_graphs += num_graphs_in_batch
                
                acc = 100 - torch.abs((data.K.to(device) - pred) / data.K.to(device)) * 100
                pred_whole.append(pred.cpu().numpy())
                K_whole.append(data.K.cpu().numpy())
                test_acc.append(acc.detach().cpu().numpy())
            
            pred_whole = np.concatenate(pred_whole, axis=0)
            K_whole = np.concatenate(K_whole, axis=0)  
            R2_test = r2_score(K_whole, pred_whole)
            R2_tests.append(R2_test)
            
            test_loss = test_loss / total_graphs
            test_losses.append(test_loss)  
            test_acc_total.append(np.mean(np.concatenate(test_acc, axis=0)))
        
        if epoch_loss < best_loss:
            best_loss = test_loss
            best_state_dict = model.state_dict()            
        print(f'Epoch [{epoch+1}], Loss[Train: {epoch_loss:.3f}, Test: {test_loss:.3f}], R2[Train: {R2_train:.3f}, Test: {R2_test:.3f}], Acc[Train: {np.mean(train_acc_total):.2f}, Test: {np.mean(test_acc_total):.2f}]')
    
    print(f'Final Test Loss: {test_losses[-1]:.4f}')
    
    return train_losses, test_losses, R2_trainings, R2_tests, train_acc_total, test_acc_total, best_state_dict
