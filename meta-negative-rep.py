import argparse
from typing import Literal

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
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--epochs', default=1000, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--temp', default=0.5, type=int)
parser.add_argument('--download', default=False, type=bool)

args = parser.parse_args()

# torch settings
torch.manual_seed(816)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# modification of https://github.com/xjtushujun/meta-weight-net

# build model
def build_model():
    model = resnet50()
    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True
    
    return model

# train
def train(train_loader, train_meta_loader, model, optim_model, teacher, optim_teacher, lr, temperature, device):
    train_loss = 0
    meta_loss = 0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    aug_transform = transforms.Compose([
        transforms.RandomCrop(224),
        normalize,
    ])

    for (x, _), (x_meta, _) in zip(train_loader, train_meta_loader):
            # settings
            model.train()
            x = x.to(device)
            x_meta = x_meta.to(device)

            x1 = aug_transform(x)
            x2 = aug_transform(x)
            x_meta1 = aug_transform(x_meta)
            x_meta2 = aug_transform(x_meta)

            model_meta = build_model().cuda()
            model_meta.load_state_dict(model.state_dict())

            # pseudo update model_meta
            rep1, _ = model_meta(x1)
            rep1 = F.normalize(rep1, p=2, dim=1)
            rep2, _ = model_meta(x2)
            rep2 = F.normalize(rep2, p=2, dim=1)

            p_rep1, _ = teacher(x1)
            p_rep2, _ = teacher(x2)
            p_rep = torch.stack((p_rep1, p_rep2), dim=1)
            p_rep = F.normalize(p_rep, p=2, dim=2)

            loss_pos = torch.exp(torch.sum(rep1 * rep2, dim=-1) / temperature)
            rep = torch.stack((rep1, rep2), dim=1)
            loss_neg_matrix = torch.exp(torch.mm(rep, p_rep.t().contiguous()) / temperature)
            loss_neg = loss_neg_matrix.view(loss_neg_matrix.size(0), -1).sum(dim=-1) # not negative samples but pseudo negative samples
            loss_p = (- torch.log(loss_pos / loss_neg)).mean()

            model_meta.zero_grad()
            grads = torch.autograd.grad(loss_p, (model_meta.params()), create_graph=True)
            model_meta.update_params(lr_inner=lr, source_params=grads)
            del grads

            # meta update teacher
            meta_rep1, _ = model_meta(x_meta1)
            meta_rep1 = F.normalize(meta_rep1, p=2, dim=1)
            meta_rep2, _ = model_meta(x_meta2)
            meta_rep2 = F.normalize(meta_rep2, p=2, dim=1)
            
            loss_meta = (-torch.sum(meta_rep1 * meta_rep2, dim=-1) / temperature).mean()

            optim_teacher.zero_grad()
            loss_meta.backward()
            optim_teacher.step()

            # update model
            rep1, _ = model_meta(x1)
            rep1 = F.normalize(rep1, p=2, dim=1)
            rep2, _ = model_meta(x2)
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

    return train_loss, meta_loss


def test(model, test_loader, device):
    # to edit
    model.eval()
    linear = nn.Linear(1024,1000)

    optimizer = optim.Adam(linear, lr=1e-3, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss().to(device)

    i = 0
    best_acc = -1.0
    while i!=3:
        for i, (x, y) in enumerate(test_loader):
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
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                rep, _ = model(x)
                rep = F.normalize(rep, p=2, dim=1)
                y_est = linear(rep)

                predicted = y_est.max(1)
                correct+=predicted.eq(y).sum().item()
        
            acc = 100. * correct / len(test_loader.dataset)
        
        if acc>=best_acc:
            best_acc = acc
            i = 0
        else:
            i+=1

    print(f"Linear Accuracy: {best_acc}")

    return best_acc


train_loader, train_meta_loader, test_loader = build_dataset(download=args.download)
model = build_model("student")
teacher = build_model("teacher")

optim_model = torch.optim.SGD(model.params(), args.lr, momentum=0.9, weight_decay=1e-6)
optim_teacher = torch.optim.Adam(teacher.params(), args.lr, weight_decay=1e-6)

def main(device):
    wandb.init(project="MetaRL", entity="ebkim")
    wandb.config = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "temperature": args.temp,
        "rep_dim": 1024
    }
    final_acc = -1.0
    for epoch in range(args.epochs):
        train_loss, meta_loss = train(train_loader, train_meta_loader, model, optim_model, teacher, optim_teacher, args.lr, args.temp, device)
        if (epoch+1)%5==0:
            print(f"Epoch: [{epoch}/{args.epochs}]\t Iters: [{i}]\t Loss: [{(train_loss/(epoch+1))}]\t MetaLoss: [{(meta_loss/(epoch+1))}]")
            train_loss = 0
            meta_loss = 0
            
            test_acc = test(model=model, test_loader=test_loader, device=device)
            if test_acc>=final_acc:
                final_acc = test_acc

            wandb.log({
                "train loss": train_loss,
                "meta loss": meta_loss,
                "accuracy": test_acc
            })

    print(f"test accuracy: {final_acc}")

if __name__=="__main__":
    main()