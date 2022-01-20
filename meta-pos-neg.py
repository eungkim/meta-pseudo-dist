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

from models.models_imagenet import resnet18, resnet50, Teacher
from models.models_cifar10 import resnet32, dresnet32
from dataset import build_dataset
from utils import adjust_learning_rate, calcul_loss, calcul_meta_loss

import wandb


# args
parser = argparse.ArgumentParser(description='Pytorch Implementation of Neural Pacer Training')
parser.add_argument('--dataset', default="cifar10", type=str)
parser.add_argument('--epochs', default=800, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--latent', default=64, type=int)
parser.add_argument('--loss', default="ntxent", type=str)
parser.add_argument('--lr', default=5e-2, type=float)
parser.add_argument('--w_decay', default=1e-4, type=float)
parser.add_argument('--temp', default=0.5, type=float)
parser.add_argument('--path', default="/home/", type=str)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--linear_lr', default=30.0, type=float)
parser.add_argument('--linear_schedule', default=[60, 80], type=int)

args = parser.parse_args()

# torch settings
torch.manual_seed(816)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# modification of https://github.com/xjtushujun/meta-weight-net


# train
def train(train_loader, train_meta_loader, model, optim_model, teacher, optim_teacher, p_lr, device):
    p_train_loss = 0
    train_loss = 0
    meta_loss = 0

    for (x1, x2), (x_meta1, x_meta2) in zip(train_loader, train_meta_loader):
        # settings
        model.train()
        x1 = x1.to(device)
        x2 = x2.to(device)

        p_model = dresnet32(args.latent)
        p_model = p_model.to(device)
        p_model.load_state_dict(model.state_dict())
        p_model.train()

        # pseudo update model_meta
        p_rep1, p_rep2 = p_model(x1, x2)
        pn_rep1 = teacher(x1)
        pn_rep2 = teacher(x2)

        loss_p = calcul_loss(p_rep1, p_rep2, pn_rep1, pn_rep2, args)
        p_grads = torch.autograd.grad(loss_p, (p_model.parameters()), create_graph=True)

        p_optim_model = MetaSGD(p_model, p_model.parameters(), lr=p_lr)
        p_optim_model.load_state_dict(optim_model.state_dict())
        p_optim_model.meta_step(p_grads)

        del p_grads

        x_meta1 = x_meta1.to(device)
        x_meta2 = x_meta2.to(device)

        # meta update teacher
        meta_rep1, meta_rep2 = p_model(x_meta1, x_meta2)
        meta_rep1 = F.normalize(meta_rep1, p=2, dim=1)
        meta_rep2 = F.normalize(meta_rep2, p=2, dim=1)

        loss_meta = (- torch.sum(meta_rep1 * meta_rep2, dim=-1)).mean()
        # meta_n_rep1 = teacher(x_meta1)
        # meta_n_rep1 = F.normalize(meta_n_rep1, p=2, dim=1)
        # meta_n_rep2 = teacher(x_meta2)
        # meta_n_rep2 = F.normalize(meta_n_rep2, p=2, dim=1)
        # loss_meta = calcul_meta_loss(meta_rep1, meta_rep2, meta_n_rep1, meta_n_rep2, args)

        optim_teacher.zero_grad()
        loss_meta.backward()
        optim_teacher.step()

        # update model
        rep1, rep2 = model(x1, x2)
        rep1 = F.normalize(rep1, p=2, dim=1)
        rep2 = F.normalize(rep2, p=2, dim=1)

        with torch.no_grad():
            n_rep1 = teacher(x1)
            n_rep2 = teacher(x2)

        loss = calcul_loss(rep1, rep2, n_rep1, n_rep2, args)

        optim_model.zero_grad()
        loss.backward()
        optim_model.step()

        # print loss
        p_train_loss += loss_p.item()
        train_loss += loss.item()
        meta_loss += loss_meta.item()

    iter_num = len(train_loader.dataset)

    return p_train_loss/iter_num, train_loss/iter_num, meta_loss/iter_num


def test(model, train_loader, valid_loader, device):
    # to edit
    model.eval()
    if args.dataset=="cifar10":
        linear = nn.Linear(args.latent,10)
    elif args.dataset=="imagenet":
        linear = nn.Linear(1024, 1000)

    linear = linear.to(device)
    optimizer = optim.SGD(linear.parameters(), lr=args.linear_lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)

    for i in range(90):
        adjust_learning_rate(optimizer, i, args)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                rep, _ = model(x, x)
                rep = F.normalize(rep, p=2, dim=1)
            y_est = linear(rep)
            loss = criterion(y_est, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    correct = 0
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            rep, _ = model(x, x)
            rep = F.normalize(rep, p=2, dim=1)
            y_est = linear(rep)

            _, predicted = y_est.max(1)
            correct+=predicted.eq(y).sum().item()
    
        train_acc = 100. * correct / len(train_loader.dataset)
    
    correct = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            rep = model(x)
            rep = F.normalize(rep, p=2, dim=1)
            y_est = linear(rep)

            _, predicted = y_est.max(1)
            correct+=predicted.eq(y).sum().item()
    
        valid_acc = 100. * correct / len(valid_loader.dataset)

    return train_acc, valid_acc


def main(device):
    wandb.init(project="MetaRL", entity="ebkim")
    wandb.config = {
        "dataset": args.dataset,
        "loss": args.loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.w_decay,
        "temperature": args.temp,
        "latent_dim": args.latent
    }
    best_valid_acc = -1.0

    model = dresnet32(args.latent)
    model = model.to(device)

    # teacher = Teacher()
    teacher = resnet32(args.latent)
    teacher = teacher.to(device)

    train_loader, train_meta_loader, train_acc_loader, valid_loader = build_dataset(type_dataset=args.dataset, batch_size=args.batch_size, num_worker=4, path=args.path)

    optim_model = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.w_decay)
    optim_teacher = torch.optim.SGD(teacher.parameters(), args.lr, momentum=0.9, weight_decay=args.w_decay)
    scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(optim_model, T_max=args.epochs, eta_min=0)
    scheduler_teacher = torch.optim.lr_scheduler.CosineAnnealingLR(optim_teacher, T_max=args.epochs, eta_min=0)

    for epoch in range(args.epochs):
        p_train_loss, train_loss, meta_loss = train(train_loader, train_meta_loader, model, optim_model, teacher, optim_teacher, scheduler_model.get_last_lr()[0], device)
        scheduler_model.step()
        scheduler_teacher.step()
        # if (epoch+1)%5==0:
        print(f"Epoch: [{epoch}/{args.epochs}]\t Pseudo Loss: [{p_train_loss}]\t Loss: [{train_loss}]\t MetaLoss: [{meta_loss}]")
        
        if ((epoch+1)%50)==0:
            train_acc, valid_acc = test(model=model, train_loader=train_acc_loader, valid_loader=valid_loader, device=device)
            print(f"Epoch: [{epoch}/{args.epochs}]\t Train Accuracy: [{train_acc}]\t Valid Accuracy: [{valid_acc}]")
            if valid_acc>=best_valid_acc:
                best_valid_acc = valid_acc 
                torch.save(model.state_dict(), f"saved_models/epoch{epoch}_b{args.batch_size}_dim{args.latent}.pth")

            wandb.log({
                "epoch": epoch,
                "pseudo train loss": p_train_loss,
                "train loss": train_loss,
                "meta loss": meta_loss,
                "train accuracy": train_acc, 
                "test accuracy": valid_acc 
            })

    print(f"Best Valid Accuracy: {best_valid_acc}")

if __name__=="__main__":
    main(device)