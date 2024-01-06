import argparse
import utils
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.distributed as dist
from model import ToyModel
from dataset import MyDataset
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm


def int2bool(v):
    """
    integer to boolean
    """
    if v == 0:
        return False
    else:
        return True


def synchronize_between_processes(t):
    t = torch.tensor(t, dtype=torch.float64, device='cuda')
    dist.barrier()
    dist.all_reduce(t)

    return t.item()


def get_args_parser():
    parser = argparse.ArgumentParser('DDP Example', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=100, type=int)

    return parser


def train(loader_train, loader_val, model, device, epochs, rank):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           factor=0.1,
                                                           patience=1,
                                                           min_lr=1e-6,
                                                           verbose=True)
    best_loss = 1e9
    for epoch in range(epochs):
        loader_train.sampler.set_epoch(epoch)
        train_loss = train_one_epoch(loader_train, model, optimizer, criterion, device, rank)
        val_loss = val_one_epoch(loader_val, model, criterion, device, rank)

        if val_loss < best_loss:
            best_loss = val_loss
            utils.save_on_master(model.module.state_dict(), "best_model.pth")

        scheduler.step(val_loss)


def train_one_epoch(loader_train, model, optimizer, criterion, device, rank):
    model.train()
    train_loss = 0.
    length = 0
    train_loop = tqdm(enumerate(loader_train, 0), total=len(loader_train), colour='blue', disable=int2bool(rank))
    for i, (x, y) in train_loop:
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)

        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        num = y.size(0)
        train_loss += loss.item() * num
        length += num

    length = synchronize_between_processes(length)
    train_loss = synchronize_between_processes(train_loss)

    avg_train_loss = train_loss / length

    print(f"train_loss: {avg_train_loss:.4f}")
    return avg_train_loss


def val_one_epoch(loader_val, model, criterion, device, rank):
    model.eval()
    valid_loss = 0.
    length = 0
    with torch.no_grad():
        val_loop = tqdm(enumerate(loader_val, 0), total=len(loader_val), colour='white', disable=int2bool(rank))
        for i, (x, y) in val_loop:
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y)

            num = y.size(0)
            valid_loss += loss.item() * num
            length += num

    length = synchronize_between_processes(length)
    valid_loss = synchronize_between_processes(valid_loss)

    avg_val_loss = valid_loss / length

    print(f"val_loss: {avg_val_loss:.4f}")
    return avg_val_loss


def test(loader_test, model, device, rank):
    model.eval()
    y_true = tuple()
    y_pred = tuple()
    test_loop = tqdm(enumerate(loader_test, 0), total=len(loader_test), colour="yellow", disable=int2bool(rank))

    with torch.no_grad():
        for i, (x, y) in test_loop:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)

            y_pred += (y.cpu().numpy(), )
            y_true += (y_hat.cpu().numpy(), )

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    np.save("y_pred.npy", y_pred)
    np.save("y_true.npy", y_true)


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device("cuda")

    seed = 666 + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    train_set = MyDataset(x=np.arange(10), y=np.arange(10))
    train_sampler = DistributedSampler(train_set, num_replicas=num_tasks,
                                       rank=global_rank, shuffle=True, seed=666)
    train_loader = DataLoader(train_set, sampler=train_sampler,
                              batch_size=args.batch_size,
                              num_workers=2)

    val_set = MyDataset(x=np.arange(10), y=np.arange(10))
    val_sampler = DistributedSampler(val_set, num_replicas=num_tasks,
                                     rank=global_rank, shuffle=False)
    val_loader = DataLoader(val_set, sampler=val_sampler,
                            batch_size=args.batch_size,
                            num_workers=2)

    test_set = MyDataset(x=np.arange(10), y=np.arange(10))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=2, shuffle=False)

    model = ToyModel()

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    print("=====> start training ...")
    train(train_loader, val_loader, model, device, args.epochs, global_rank)

    if global_rank == 0:
        print("=====> start testing ...")
        ckpt = torch.load("best_model.pth", map_location="cpu")  # cpu is better
        test_model = ToyModel().to(device)
        test_model.load_state_dict(ckpt, strict=False)
        test(test_loader, test_model, device, global_rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DDP Example script', parents=[get_args_parser()])
    my_args = parser.parse_args()
    main(my_args)
