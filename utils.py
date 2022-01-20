import torch
import torch.nn.functional as F


# from https://github.com/facebookresearch/moco
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.linear_lr
    for milestone in args.linear_schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def calcul_loss(rep1, rep2, n_rep1, n_rep2, args):
    rep1 = F.normalize(rep1, p=2, dim=1) 
    rep2 = F.normalize(rep2, p=2, dim=1)
    n_rep1 = F.normalize(n_rep1, p=2, dim=1)
    n_rep2 = F.normalize(n_rep2, p=2, dim=1)

    loss_pos = torch.sum(rep1 * rep2, dim=-1) / args.temp

    if args.loss=="ntxent":
        p_rep = torch.stack((n_rep1, n_rep2), dim=2)
        rep = torch.stack((rep1, rep2), dim=1)
        loss_neg_matrix = torch.exp(torch.matmul(rep, p_rep) / args.temp)
        loss_neg = loss_neg_matrix.view(loss_neg_matrix.size(0), -1).sum(dim=-1) # not negative samples but pseudo negative samples
        loss_p = (-loss_pos + torch.log(loss_neg)).mean()

    elif args.loss=="npair":
        loss_neg1 = torch.sum(rep1 * n_rep1, dim=-1) / args.temp
        loss_neg2 = torch.sum(rep2 * n_rep2, dim=-1) / args.temp
        loss_p = (-loss_pos + torch.log(torch.exp(loss_pos) + torch.exp(loss_neg1) + torch.exp(loss_neg2))).mean()

    else:
        print("error")

    return loss_p

def calcul_multi_neg_loss(rep1, rep2, n_rep1, n_rep2, args):
    rep1 = F.normalize(rep1, p=2, dim=1)
    rep2 = F.normalize(rep2, p=2, dim=1)
    n_rep1 = F.normalize(n_rep1, p=2, dim=1)
    n_rep2 = F.normalize(n_rep2, p=2, dim=1)

    loss_pos = torch.sum(rep1 * rep2, dim=-1) / args.temp

    if args.loss=="uni":
        rep1 = rep1.unsqueeze(dim=1)
        rep2 = rep2.unsqueeze(dim=1)
        loss_neg1_matrix = torch.exp(torch.matmul(rep1, n_rep1) / args.temp)
        loss_neg1 = loss_neg1_matrix.view(loss_neg1_matrix.size(0), -1).sum(dim=-1)
        loss_neg2_matrix = torch.exp(torch.matmul(rep2, n_rep2) / args.temp)
        loss_neg2 = loss_neg2_matrix.view(loss_neg2_matrix.size(0), -1).sum(dim=-1)

        loss_p = (-loss_pos + torch.log(loss_neg1 + loss_neg2)).mean()

    elif args.loss=="cross":
        pn_rep = torch.cat((n_rep1, n_rep2), dim=2)
        rep = torch.stack((rep1, rep2), dim=1)
        loss_neg_matrix = torch.exp(torch.matmul(rep, pn_rep) / args.temp)
        loss_neg = loss_neg_matrix.view(loss_neg_matrix.size(0), -1).sum(dim=-1) # not negative samples but pseudo negative samples
        loss_p = (-loss_pos + torch.log(loss_neg)).mean()

    return loss_p


def calcul_meta_loss(rep1, rep2, args):
    rep1 = F.normalize(rep1, p=2, dim=1) 
    rep2 = F.normalize(rep2, p=2, dim=1)

    loss_pos1 = torch.sum(rep1 * rep2, dim=-1) / args.temp
    loss_p = (-torch.log(torch.exp(loss_pos1)+torch.exp(loss_pos2)+torch.exp(loss_pos3)) + torch.log(torch.exp(loss_neg1)+torch.exp(loss_neg2))).mean()

    return loss_p