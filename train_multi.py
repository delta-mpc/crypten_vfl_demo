import sys

import crypten
import crypten.communicator as comm
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data_utils import crypten_collate
from model import MLP

names = ["a", "b", "c"]
feature_sizes = [50, 57, 1]


def load_local_tensor(filename: str) -> torch.Tensor:
    arr = np.load(filename)
    if filename.endswith(".npz"):
        arr = arr["arr_0"]
    tensor = torch.tensor(arr, dtype=torch.float32)
    return tensor


def load_encrypt_tensor(filename: str) -> crypten.CrypTensor:
    local_tensor = load_local_tensor(filename)
    rank = comm.get().get_rank()
    count = local_tensor.shape[0]

    encrypt_tensors = []
    for i, (name, feature_size) in enumerate(zip(names, feature_sizes)):
        if rank == i:
            assert local_tensor.shape[1] == feature_size, \
                f"{name} feature size should be {feature_size}, but get {local_tensor.shape[1]}"
            tensor = crypten.cryptensor(local_tensor, src=i)
        else:
            dummy_tensor = torch.zeros((count, feature_size), dtype=torch.float32)
            tensor = crypten.cryptensor(dummy_tensor, src=i)
        encrypt_tensors.append(tensor)

    res = crypten.cat(encrypt_tensors, dim=1)
    return res


def make_local_dataloader(filename: str, batch_size: int, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
    tensor = load_local_tensor(filename)
    dataset = TensorDataset(tensor)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


def make_mpc_model(local_model: torch.nn.Module):
    dummy_input = torch.empty((1, 107))
    model = crypten.nn.from_pytorch(local_model, dummy_input)
    model.encrypt()
    return model


def make_mpc_dataloader(filename: str, batch_size: int, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
    mpc_tensor = load_encrypt_tensor(filename)
    feature, label = mpc_tensor[:, :-1], mpc_tensor[:, -1]
    dataset = TensorDataset(feature, label)
    seed = (crypten.mpc.MPCTensor.rand(1) * (2 ** 32)).get_plain_text().int().item()
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=drop_last,
                            collate_fn=crypten_collate, generator=generator)
    return dataloader


def train_mpc(dataloader: DataLoader, model: crypten.nn.Module, loss: crypten.nn.Module, lr: float):
    total_loss = None
    count = len(dataloader)

    model.train()
    for xs, ys in tqdm(dataloader, file=sys.stdout):
        out = model(xs)
        loss_val = loss(out, ys)

        model.zero_grad()
        loss_val.backward()
        model.update_parameters(lr)

        if total_loss is None:
            total_loss = loss_val.detach()
        else:
            total_loss += loss_val.detach()

    total_loss = total_loss.get_plain_text().item()
    return total_loss / count


def validate_mpc(dataloader: DataLoader, model: crypten.nn.Module, loss: crypten.nn.Module):
    model.eval()
    outs = []
    true_ys = []
    total_loss = None
    count = len(dataloader)
    for xs, ys in tqdm(dataloader, file=sys.stdout):
        out = model(xs)
        loss_val = loss(out, ys)

        outs.append(out)
        true_ys.append(ys)

        if total_loss is None:
            total_loss = loss_val.detach()
        else:
            total_loss += loss_val.detach()

    total_loss = total_loss.get_plain_text().item()

    all_out = crypten.cat(outs, dim=0)
    all_prob = all_out.sigmoid()
    all_prob = all_prob.get_plain_text()
    pred_ys = torch.where(all_prob > 0.5, 1, 0).tolist()
    pred_probs = all_prob.tolist()

    true_ys = crypten.cat(true_ys, dim=0)
    true_ys = true_ys.get_plain_text().tolist()

    return total_loss / count, precision_score(true_ys, pred_ys), recall_score(true_ys, pred_ys), \
           roc_auc_score(true_ys, pred_probs)


def test():
    crypten.init()
    rank = comm.get().get_rank()

    name = names[rank]
    filename = f"dataset/{name}/train.npz"

    mpc_tensor = load_encrypt_tensor(filename)
    feature, label = mpc_tensor[:32, :-1], mpc_tensor[:32, -1]
    print(feature.shape, feature.ptype)

    model = MLP()
    mpc_model = make_mpc_model(model)
    loss = crypten.nn.BCELoss()

    mpc_model.train()
    out = mpc_model(feature)
    prob = out.sigmoid()
    loss_val = loss(prob, label)

    mpc_model.zero_grad()
    loss_val.backward()
    mpc_model.update_parameters(1e-3)


def main():
    epochs = 50
    batch_size = 32
    lr = 1e-3
    eval_every = 1

    crypten.init()
    rank = comm.get().get_rank()

    name = names[rank]
    train_filename = f"dataset/{name}/train.npz"
    test_filename = f"dataset/{name}/test.npz"

    train_dataloader = make_mpc_dataloader(train_filename, batch_size, shuffle=True, drop_last=False)
    test_dataloader = make_mpc_dataloader(test_filename, batch_size, shuffle=False, drop_last=False)

    model = MLP()
    mpc_model = make_mpc_model(model)
    mpc_loss = crypten.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        train_loss = train_mpc(train_dataloader, mpc_model, mpc_loss, lr)
        print(f"epoch: {epoch}, train loss: {train_loss}")

        if epoch % eval_every == 0:
            validate_loss, p, r, auc = validate_mpc(test_dataloader, mpc_model, mpc_loss)
            print(f"epoch: {epoch}, validate loss: {validate_loss}, precision: {p}, recall: {r}, auc: {auc}")


if __name__ == '__main__':
    main()
