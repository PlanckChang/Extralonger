import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, get_index
# ! X shape: (B, T, N, C)

def get_scaler(train_dataset, batch_size):
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    mean_ls = []
    std_ls = []
    for x, _ in train_dataloader:
        mean = x[..., 0].mean() 
        mean_ls.append(mean.item())
    mean = np.array(mean_ls).mean()
    for x, _ in train_dataloader:
        std = (x[..., 0] - mean).pow(2).mean()    
        std_ls.append(std.item())
    std = np.array(std_ls).mean() ** 0.5   
    scaler = StandardScaler(mean, std) 
    return scaler    
    
class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, data, index, scaler = None):
        self.data = data
        self.index = index
        self.scaler = scaler
        
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        x_start, x_stop, y_start, y_stop = self.index[idx]
        x = self.data[x_start : x_stop, :, :].copy()
        y = self.data[y_start : y_stop, :, :1]
        if self.scaler:
            x[..., 0] = self.scaler.transform(x[..., 0])
        return torch.FloatTensor(x), torch.FloatTensor(y)
    
    def get_x_shape(self):
        return (len(self),) + self[0][0].shape

    def get_y_shape(self):
        return (len(self),) + self[0][1].shape

def get_dataloaders(
    data_dir, batch_size=64, log=None, in_steps=288,  out_steps=288
):
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)
    train_index, val_index, test_index = get_index(in_steps, data.shape[0],  out_steps)
    train_dataset = IndexDataset(data, train_index)
    scaler = get_scaler(train_dataset, batch_size)
    train_dataset.scaler = scaler
    val_dataset = IndexDataset(data, val_index, scaler)
    test_dataset = IndexDataset(data, test_index, scaler)
     
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    
    print_log(f"Trainset:\tx-{train_dataset.get_x_shape()}\ty-{train_dataset.get_y_shape()}", log=log)
    print_log(f"Valset:  \tx-{val_dataset.get_x_shape()}  \ty-{val_dataset.get_y_shape()}", log=log)
    print_log(f"Testset:\tx-{test_dataset.get_x_shape()}\ty-{test_dataset.get_y_shape()}", log=log)

    return train_loader, val_loader, test_loader, scaler