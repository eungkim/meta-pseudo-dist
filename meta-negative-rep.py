import argparse
import os
from typing import Literal
from meta import MetaSGD

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np

from models import resnet50
from dataset import build_dataset

import wandb


# args
parser = argparse.ArgumentParser(description='Pytorch Implementation of Neural Pacer Training')
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=5e-2, type=float)
parser.add_argument('--w_decay', default=1e-4, type=float)
parser.add_argument('--temp', default=0.5, type=float)
parser.add_argument('--path', default="/home/", type=str)

args = parser.parse_args()

# torch settings
torch.manual_seed(816)
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# modification of https://github.com/xjtushujun/meta-weight-net


# train
def train(train_loader, train_meta_loader, model, optim_model, teacher, optim_teacher, temperature, device):
    train_loss = 0
    meta_loss = 0

    for (x1, x2), (x_meta1, x_meta2) in zip(train_loader, train_meta_loader):
        # settings
        model.train()
        x1 = x1.to(device)
        x2 = x2.to(device)

        # p_model = resnet50().to(device)
        p_model = resnet50().to(device)
        p_model = nn.DataParallel(p_model)
        p_model.cuda()
        p_model.load_state_dict(model.state_dict())
        p_model.train()

        # pseudo update model_meta
        rep1, _ = p_model(x1)
        rep1 = F.normalize(rep1, p=2, dim=1)
        rep2, _ = p_model(x2)
        rep2 = F.normalize(rep2, p=2, dim=1)

        p_rep1, _ = teacher(x1)
        p_rep2, _ = teacher(x2)
        p_rep = torch.stack((p_rep1, p_rep2), dim=2)
        p_rep = F.normalize(p_rep, p=2, dim=1)

        loss_pos = torch.exp(torch.sum(rep1 * rep2, dim=-1) / temperature)
        rep = torch.stack((rep1, rep2), dim=1)
        loss_neg_matrix = torch.exp(torch.matmul(rep, p_rep) / temperature)
        loss_neg = loss_neg_matrix.view(loss_neg_matrix.size(0), -1).sum(dim=-1) # not negative samples but pseudo negative samples
        loss_p = (- torch.log(loss_pos / loss_neg)).mean()

        grads = torch.autograd.grad(loss_p, (p_model.parameters()), create_graph=True)
        p_optim_model = MetaSGD(p_model, p_model.parameters(), lr=scheduler_model.get_last_lr()[0], create_graph=True)
        p_optim_model.load_state_dict(optim_model.state_dict())
        p_optim_model.meta_step(grads)

        del grads
        print("breakpoint1")

        x_meta1 = x_meta1.to(device)
        x_meta2 = x_meta2.to(device)
        # meta update teacher
        meta_rep1, _ = p_model(x_meta1)
        meta_rep1 = F.normalize(meta_rep1, p=2, dim=1)
        meta_rep2, _ = p_model(x_meta2)
        meta_rep2 = F.normalize(meta_rep2, p=2, dim=1)
        
        loss_meta = (-torch.sum(meta_rep1 * meta_rep2, dim=-1) / temperature).mean()
        print("breakpoint2")

        optim_teacher.zero_grad()
        loss_meta.backward()
        optim_teacher.step()
        print("breakpoint3")

        # update model
        rep1, _ = model(x1)
        rep1 = F.normalize(rep1, p=2, dim=1)
        rep2, _ = model(x2)
        rep2 = F.normalize(rep2, p=2, dim=1)

        with torch.no_grad():
            p_rep1, _ = teacher(x1)
            p_rep2, _ = teacher(x2)
            p_rep = torch.stack((p_rep1, p_rep2), dim=1)
            p_rep = F.normalize(p_rep, p=2, dim=2)

        loss_pos = torch.exp(torch.sum(rep1 * rep2, dim=-1) / temperature)
        rep = torch.stack((rep1, rep2), dim=1)
        loss_neg_matrix = torch.exp(torch.mm(rep, p_rep.t().contiguous()) / temperature)
        loss_neg = loss_neg_matrix.view(loss_neg_matrix.size(0), -1).sum(dim=-1) # not negative samples but pseudo negative samples
        loss_p = (- torch.log(loss_pos / loss_neg)).mean()
        
        optim_model.zero_grad()
        loss_p.backward()
        optim_model.step()

        # print loss
        train_loss += loss_p.item()
        meta_loss += loss_meta.item()

    iter_num = len(train_loader.dataset)

    return train_loss/iter_num, meta_loss/iter_num


def test(model, valid_loader, device):
    # to edit
    model.eval()
    linear = nn.Linear(1024,1000)

    optimizer = optim.Adam(linear, lr=1e-3, weight_decay=args.w_decay)
    criterion = nn.CrossEntropyLoss().to(device)

    i = 0
    best_acc = -1.0
    while i!=3:
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                rep, _ = model(x)
                rep = F.normalize(rep, p=2, dim=1)
            y_est = linear(rep)
            loss = criterion(y_est, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        correct = 0

        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                rep, _ = model(x)
                rep = F.normalize(rep, p=2, dim=1)
                y_est = linear(rep)

                predicted = y_est.max(1)
                correct+=predicted.eq(y).sum().item()
        
            acc = 100. * correct / len(valid_loader.dataset)
        
        if acc>=best_acc:
            best_acc = acc
            i = 0
        else:
            i+=1

    return best_acc


train_loader, train_meta_loader, train_acc_loader, valid_loader = build_dataset(batch_size=args.batch_size, path=args.path)

model = resnet50()
model = nn.DataParallel(model)
model.cuda()

teacher = resnet50()
teacher = nn.DataParallel(teacher)
teacher.cuda()

optim_model = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.w_decay)
optim_teacher = torch.optim.SGD(teacher.parameters(), args.lr, momentum=0.9, weight_decay=args.w_decay)
scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(optim_model, T_max=args.epochs, eta_min=0)
scheduler_teacher = torch.optim.lr_scheduler.CosineAnnealingLR(optim_teacher, T_max=args.epochs, eta_min=0)


def main(device):
    wandb.init(project="MetaRL", entity="ebkim")
    wandb.config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.w_decay,
        "temperature": args.temp,
        "rep_dim": 1024
    }

    best_train_acc = -1.0
    best_valid_acc = -1.0
    for epoch in range(args.epochs):
        train_loss, meta_loss = train(train_loader, train_meta_loader, model, optim_model, teacher, optim_teacher, args.temp, device)
        scheduler_model.step()
        scheduler_teacher.step()
        # if (epoch+1)%5==0:
        print(f"Epoch: [{epoch}/{args.epochs}]\t Loss: [{train_loss}]\t MetaLoss: [{meta_loss}]")
        
        train_acc = test(model=model, valid_loader=train_acc_loader, device=device)
        print(f"Epoch: [{epoch}/{args.epochs}]\t Train Accuracy: [{train_acc}]")
        if train_acc>=best_train_acc:
            best_train_acc = train_acc 
        
        valid_acc = test(model=model, valid_loader=valid_loader, device=device)
        print(f"Epoch: [{epoch}/{args.epochs}]\t Valid Accuracy: [{valid_acc}]")
        if valid_acc>=best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), f"saved_models/epoch{epoch}.pth")

        wandb.log({
            "train loss": train_loss,
            "meta loss": meta_loss,
            "train accuracy": train_acc, 
            "test accuracy": valid_acc 
        })

    print(f"Best Valid Accuracy: {best_valid_acc}")

if __name__=="__main__":
    main(device)