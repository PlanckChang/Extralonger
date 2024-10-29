import argparse
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import datetime
import time
from torchinfo import summary
import yaml
import json
import sys
import copy

from lib.utils import (
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)
from lib.metrics import MSE_MAE_MAPE
from lib.data_prepare import get_dataloaders
from model.Extralonger import Extralonger

@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())
    return np.mean(batch_loss_list)

@torch.no_grad()
def predict(model, loader):
    model.eval()
    predict_result = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        predict_result.append(MSE_MAE_MAPE(y_batch, out_batch))
    predict_result = np.mean(predict_result, axis=0)
    predict_result[0] = np.sqrt(predict_result[0]) 
    return predict_result

def train_one_epoch(
    model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=None
):
    global cfg, global_iter_count, global_target_length

    model.train()
    batch_loss_list = []
    for x_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)        
        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
    epoch_loss = np.mean(batch_loss_list)
    return epoch_loss


def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    scheduler,
    criterion,
    clip_grad=0,
    max_epochs=200,
    early_stop=10,
    verbose=1,
    log=None,
    save=None,
):
    model = model.to(DEVICE)
    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []
    start_train = time.time()
    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=log
        )
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, valset_loader, criterion)
        val_loss_list.append(val_loss)
        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                " \tTrain Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                log=log,
            )
        
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= early_stop:
                break
    end_train = time.time()    
    train_time = datetime.timedelta(seconds=end_train-start_train)
    model.load_state_dict(best_state_dict)

    train_rmse, train_mae, train_mape = predict(model, trainset_loader)
    val_rmse, val_mae, val_mape = predict(model, valset_loader)
    out_str = f"Train during time: {str(train_time)}\n"
    out_str += f"Early stopping at epoch: {epoch+1}\n"
    out_str += f"Best at epoch: {best_epoch+1}\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)
    if save:
        torch.save(best_state_dict, save)
    return model

@torch.no_grad()
def test_model(model, testset_loader, log=None, mode=None):
    model.eval()
    print_log("--------- Test ---------", log=log)
    start = time.time()
    rmse_all, mae_all, mape_all = predict(model, testset_loader)
    end = time.time()
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)

def get_mask(dataset='PEMS04', num_nodes=307):
    adj_matrix = os.path.join('./data', dataset, 'adj.csv')
    data = pd.read_csv(adj_matrix)
    adj_matrix = torch.eye(num_nodes, dtype=torch.bool, device=DEVICE)
    for row in data.itertuples(index=False):
        from_node, to_node, _ = row
        adj_matrix[from_node, to_node] = True
        adj_matrix[to_node, from_node] = True
    return adj_matrix

if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="pems04")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    parser.add_argument("-m", "--message", type=str, default="step 48, best params")
    parser.add_argument('-i', '--in_steps', type=int, default=36)
    parser.add_argument('-o', '--out_steps', type=int, default=36)
    parser.add_argument('--out_day', type=int, default=0)
    args = parser.parse_args()
    if args.out_day:
        args.out_steps = args.out_day * 288
    dataset = args.dataset
    dataset = dataset.upper()
    assert dataset in ["PEMS04", "PEMS08", "SEATTLE"]
    data_path = f"./data/{dataset}"
    
    with open(f"./model/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]        
    
    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}" 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg['in_steps'] = args.in_steps
    cfg['out_steps'] = args.out_steps 
    cfg['model_args']['in_steps'] = args.in_steps
    cfg['model_args']['out_steps'] = args.out_steps

    #--------------------------------- seed -----------------------------------#
            
    seed = torch.randint(1000, (1,)) # set random seed here

    seed_everything(seed)
    set_cpu_num(1)

    #--------------------------------- get mask and load model --------------------#
    adj_mask = get_mask(dataset, cfg['num_nodes'])
    model = Extralonger(**cfg["model_args"], adj_mask=adj_mask)
        
    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"./logs/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    model_name = model.__class__.__name__
    log = os.path.join(log_path, f"{model_name}-{dataset}-{now}-{args.message}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    # ------------------------------- load dataset ------------------------------- #

    print_log("dataset:", dataset, log=log)
    print_log("message:",args.message, log=log)
    print_log("gpu:", args.gpu_num, log=log)
    print_log('seed:', seed, log=log)
    print_log('in_steps', args.in_steps, log=log)
    print_log('out_steps', args.out_steps, log=log)
    print_log('model:', model_name, log=log)
 
    (
        trainset_loader,
        valset_loader, 
        testset_loader,
        SCALER,
    ) = get_dataloaders(
        data_path,
        batch_size=cfg.get("batch_size", 64),
        log=log,
        in_steps=args.in_steps,
        out_steps=args.out_steps
    )
    
    print_log(log=log)

    # --------------------------- set model saving path -------------------------- #

    save_path = f"./saved_models/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{dataset}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    criterion = nn.HuberLoss()
        
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["milestones"],
        gamma=cfg.get("lr_decay_rate", 0.1),
        verbose=False,
    )
    
    # --------------------------- print model structure -------------------------- #

    print_log("---------", model_name, "---------", log=log)
    print_log( 
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    print_log(
        summary(
            model,
            [
                cfg["batch_size"],
                cfg["in_steps"],
                cfg["num_nodes"],
                next(iter(trainset_loader))[0].shape[-1],
            ],
            verbose=0,  
        ),
        log=log,
    )
    print_log(log=log)

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)

    model = train(
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        criterion,
        clip_grad=cfg.get("clip_grad"),
        max_epochs=cfg.get("max_epochs", 200),
        early_stop=cfg.get("early_stop", 10),
        verbose=1,
        log=log,
        save=save,
    )
    
    print_log(f"Saved Model: {save}", log=log)
    test_model(model, testset_loader, log=log)
    log.close()
