import torch
    
def MSE(y_true, y_pred):
    mask = torch.not_equal(y_true, 0).float()
    mask /= torch.mean(mask)
    mse = torch.square(y_pred - y_true)
    mse = torch.mul(mse, mask)
    mse = torch.mean(mse)
    return mse

def RMSE(y_true, y_pred):
    mask = torch.not_equal(y_true, 0).float()
    mask /= torch.mean(mask)
    rmse = torch.square(torch.abs(y_pred - y_true))
    rmse = torch.mul(rmse, mask)
    rmse = torch.sqrt(torch.mean(rmse))
    return rmse

def MAE(y_true, y_pred):
    mask = torch.not_equal(y_true, 0).float()
    mask /= torch.mean(mask)
    mae = torch.abs(y_pred - y_true)
    mae = torch.mul(mae, mask)
    mae = torch.mean(mae)
    return mae

def MAPE(y_true, y_pred, null_val=0):
    if torch.isnan(torch.tensor(null_val, device=y_true.device)):
        mask = ~torch.isnan(y_true)
    else:
        mask = torch.not_equal(y_true, null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mape = torch.abs(torch.div((y_pred - y_true).float(), y_true))
    mape = torch.nan_to_num(mape)
    mape = torch.mul(mask, mape)
    return torch.mean(mape) * 100

def MSE_MAE_MAPE(y_true, y_pred):
    return (
        MSE(y_true, y_pred).item(),
        MAE(y_true, y_pred).item(),
        MAPE(y_true, y_pred).item(),
    )