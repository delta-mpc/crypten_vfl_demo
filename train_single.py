import sys

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model import MLP


def load_tensor(filename: str) -> torch.Tensor:
    arr = np.load(filename)
    if filename.endswith(".npz"):
        arr = arr["arr_0"]
    tensor = torch.tensor(arr)
    return tensor


def make_dataloader(filename: str, batch_size: int = None, shuffle: bool = False,
                    drop_last: bool = False) -> DataLoader:
    tensor = load_tensor(filename)
    feature, label = tensor[:, :-1], tensor[:, -1]
    feature = feature.float()
    label = label.float()
    dataset = TensorDataset(feature, label)
    if batch_size is None:
        batch_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


def train(dataloader: DataLoader, model: torch.nn.Module, loss: torch.nn.Module, optimizer: optim.Optimizer):
    model.train()
    count = len(dataloader)
    total_loss = 0
    for xs, ys in tqdm(dataloader, file=sys.stdout):
        out = model(xs)
        loss_val = loss(out, ys)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        total_loss += loss_val.item()
    return total_loss / count


def validate(dataloader: DataLoader, model: torch.nn.Module, loss: torch.nn.Module):
    model.eval()

    count = len(dataloader)
    total_loss = 0
    pred_probs = []
    true_ys = []
    with torch.no_grad():
        for xs, ys in tqdm(dataloader, file=sys.stdout):
            out = model(xs)
            loss_val = loss(out, ys)

            pred_prob = torch.sigmoid(out)
            pred_probs.extend(torch.flatten(pred_prob).tolist())
            true_ys.extend(torch.flatten(ys).int().tolist())
            total_loss += loss_val.item()
    pred_ys = [int(p > 0.5) for p in pred_probs]

    return total_loss / count, precision_score(true_ys, pred_ys), recall_score(true_ys, pred_ys), \
           roc_auc_score(true_ys, pred_probs)


if __name__ == '__main__':
    train_filename = "dataset/adult.train.npz"
    test_filename = "dataset/adult.test.npz"
    epochs = 50
    batch_size = 32
    lr = 1e-3
    eval_every = 1

    train_dataloader = make_dataloader(train_filename, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = make_dataloader(test_filename, batch_size=batch_size, shuffle=False, drop_last=False)

    mlp = MLP()
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(mlp.parameters(), lr)

    for epoch in range(epochs):
        train_loss = train(train_dataloader, mlp, loss, optimizer)
        print(f"epoch: {epoch}, train loss: {train_loss}")

        if epoch % eval_every == 0:
            validate_loss, p, r, auc = validate(test_dataloader, mlp, loss)
            print(f"epoch: {epoch}, validate loss: {validate_loss}, precision: {p}, recall: {r}, auc: {auc}")
