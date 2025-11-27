import torch
import torch.nn as nn
from . import pbrf_loss

def flatten_parameters(model):
    flatten_params = []
    for p in model.parameters():
        flatten_params.append(p.view(-1))
    return torch.cat(flatten_params)


def train(data, model, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    logit = model(data)
    train_loss = criterion(logit[data.train_mask], data.y[data.train_mask])

    val_loss = criterion(logit[data.val_mask], data.y[data.val_mask])
    test_loss = criterion(logit[data.test_mask], data.y[data.test_mask])

    pred = torch.argmax(logit, dim=1)

    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).to(torch.float).mean()
    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).to(torch.float).mean()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).to(torch.float).mean()

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    return train_loss, val_loss, test_loss, train_acc, val_acc, test_acc

def eval_model(data, model, device):
    model.eval()
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()

        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.edge_weight = data.edge_weight.to(device)
        data.y = data.y.to(device)

        logit = model(data)
        train_loss = criterion(logit[data.train_mask], data.y[data.train_mask])
        val_loss = criterion(logit[data.val_mask], data.y[data.val_mask])
        test_loss = criterion(logit[data.test_mask], data.y[data.test_mask])

        pred = torch.argmax(logit, dim=1)

        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).to(torch.float).mean()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).to(torch.float).mean()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).to(torch.float).mean()

    return train_loss, val_loss, test_loss, train_acc, val_acc, test_acc

def train_pbrf(influenced_nodes, data, perturbed_data, model, optimizer, device, y_s, theta_s, berman_grad, args):
    model.train()
    criterion = pbrf_loss
    loss_func = nn.CrossEntropyLoss()

    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)

    logit = model(data)
    perturbed_logit = model(perturbed_data)
    label_r = data.y[influenced_nodes]
    label = data.y
    theta = flatten_parameters(model)
    num_trains = data.train_mask.sum()

    train_loss, remove_loss, add_loss = criterion(logit[influenced_nodes], perturbed_logit[influenced_nodes], logit[data.train_mask], y_s[data.train_mask], label_r, label[data.train_mask], loss_func, args.damp, theta, theta_s, berman_grad, num_trains)

    pred = torch.argmax(logit, dim=1)

    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).to(torch.float).mean()
    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).to(torch.float).mean()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).to(torch.float).mean()

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    return train_loss, remove_loss, add_loss, train_acc, val_acc, test_acc

