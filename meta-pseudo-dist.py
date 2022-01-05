import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np

from models import ResNet32, Student
from dataset import build_dataset


# args
parser = argparse.ArgumentParser(description='Pytorch Implementation of Neural Pacer Training')
parser.add_argument('--name_dataset', default='imagenet', type=str)
parser.add_argument('--model', default="resnet", type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--dataset', default="cifar10", type=str)
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--lr', default=1e-3, type=int)

args = parser.parse_args()

# torch settings
torch.manual_seed(816)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# modification of https://github.com/xjtushujun/meta-weight-net

# build model
def build_model(name):
    if name=="student":
        model = ResNet32(args.dataset=='cifar10' and 10 or 100)
    elif name=="teacher":
        model = Student(args.dataset=='cifar10' and 10 or 100)
    else:
        print("wrong model name, choose student or teacher")

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True
    
    return model

# train
def train(train_loader, train_meta_loader, model, optim_model, teacher, optim_teacher, epochs, lr, device):
    for epoch in range(epochs):
        train_loss = 0
        meta_loss = 0
        for i, ((inputs, targets), (inputs_meta, targets_meta)) in enumerate(zip(train_loader, train_meta_loader)):
                # settings
                model.train()
                inputs = inputs.to(device)
                targets = targets.to(device)
                inputs_meta = inputs_meta.to(device)
                targets_meta = targets_meta.to(device)

                model_meta = build_model().cuda()
                model_meta.load_state_dict(model.state_dict())

                # pseudo update model_meta
                x_hat, mu, log_var = model_meta(inputs)
                _, mu_hat, log_var_hat = model_meta(x_hat.detach())
                p_mu, p_log_var = teacher(inputs) # gradient should not flow to model_meta by reps
                p_mu_hat, p_log_var_hat = teacher(x_hat.detach())

                loss_p_kl = 0.5 * torch.sum(1 + log_var - p_log_var - ((log_var.exp() + (mu - p_mu).pow(2))/p_log_var.exp()))
                loss_p_kl_hat = 0.5 * torch.sum(1 + log_var_hat - p_log_var_hat - ((log_var_hat.exp() + (mu_hat - p_mu_hat).pow(2))/p_log_var_hat.exp())) 
                loss_p = loss_p_kl + loss_p_kl_hat

                model_meta.zero_grad()
                grads = torch.autograd.grad(loss_p, (model_meta.params()), create_graph=True)
                model_meta.update_params(lr_inner=lr, source_params=grads)
                del grads

                # meta update teacher
                meta_x_hat, meta_mu, meta_log_var = model_meta(inputs_meta)            
                _, meta_mu_hat, meta_log_var_hat = model_meta(meta_x_hat.detach())

                loss_meta_kl = -0.5 * torch.sum(1 + meta_log_var - meta_log_var_hat - ((meta_log_var.exp() + (meta_mu - meta_mu_hat).pow(2))/meta_log_var_hat.exp()))

                optim_teacher.zero_grad()
                loss_meta_kl.backward()
                optim_teacher.step()

                # update model
                x_hat, mu, log_var = model(inputs)
                x_hat = x_hat.detach()
                x_til, mu_hat, log_var_hat = model(x_hat)
                with torch.no_grad():
                    p_mu, p_log_var = teacher(inputs)
                    p_mu_hat, p_log_var_hat = teacher(x_hat)
                
                loss_recon = F.binary_cross_entropy(x_hat, inputs, reduction="sum")
                loss_recon_hat = F.binary_cross_entropy(x_til, x_hat, reduction="sum")
                loss_kl = -0.5 * torch.sum(1 + log_var - log_var_hat - ((log_var.exp() + (mu - mu_hat).pow(2))/log_var_hat.exp()))
                loss_nkl = 0.5 * torch.sum(1 + log_var - p_log_var - ((log_var.exp() + (mu - p_mu).pow(2))/p_log_var.exp()))
                loss_nkl_hat = 0.5 * torch.sum(1 + log_var_hat - p_log_var_hat - ((log_var_hat.exp() + (mu_hat - p_mu_hat).pow(2))/p_log_var_hat.exp())) 
                loss = loss_recon + loss_recon_hat + loss_kl + loss_nkl + loss_nkl_hat

                optim_model.zero_grad()
                loss.backward()
                optim_model.step()

                # print loss
                train_loss += loss.item()
                meta_loss += loss_meta_kl.item()

                if (i+1)%50==0:
                    print(f"Epoch: [{epoch}/{epochs}]\t Iters: [{i}]\t Loss: [{(train_loss/(i+1))}]\t MetaLoss: [{(meta_loss/(i+1))}]")

                    train_loss = 0
                    meta_loss = 0

def test(model, test_loader):
    # to edit
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            test_loss+=F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct+=predicted.eq(targets).sum().item()
        
        test_loss/=len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

        print(f"Test Avg Loss: {test_loss} Accuracy: {accuracy}")

    return accuracy

train_loader, train_meta_loader, test_loader = build_dataset(args.name_dataset)
model = build_model("student")
teacher = build_model("teacher")

optim_model = torch.optim.SGD(model.params(), args.lr, momentum=0.9, weight_decay=1e-4)
optim_teacher = torch.optim.Adam(teacher.params(), args.lr, weight_decay=1e-4)

def main():
    best_acc = 0
    train(train_loader, train_meta_loader, model, optim_model, teacher, optim_teacher, args.epochs, args.lr, device)
    test_acc = test(model=model, test_loader=test_loader)
    if test_acc>=best_acc:
        best_acc = test_acc
    
    print(f"test accuracy: {best_acc}")


if __name__=="__main__":
    main()