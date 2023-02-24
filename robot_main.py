import argparse

import torch.optim
from meta import *
from model import Lenet, sig_t
from Resnet import ResNet18, ResNet34
from noisy_long_tail_CIFAR import *
from utils import *
import torch.nn as nn
import os
import wandb
from torch.optim.lr_scheduler import MultiStepLR
import time
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='ROBOT')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--meta_net_hidden_size', type=int, default=100)
parser.add_argument('--meta_net_num_layers', type=int, default=1)

parser.add_argument('--lr', type=float, default=.1)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--dampening', type=float, default=0.)
parser.add_argument('--nesterov', type=bool, default=False)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--meta_lr', type=float, default=1e-5)
parser.add_argument('--meta_weight_decay', type=float, default=0.)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--loss', type=str, default='ce')
parser.add_argument('--outer_obj', type=str, default='rce')
parser.add_argument('--network', type=str, default='r18')
parser.add_argument('--meta_optim', type=str, default='adam')
parser.add_argument('--proxy_input', type=str, choices=["loss", "logits", "var", "margin", "all", "out+label+add", "out+label+concat", "loss+label","loss+var","loss+std", "loss+out+label"], default='loss')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--imbalanced_factor', type=int, default=None)
parser.add_argument('--corruption_type', type=str, default=None)
parser.add_argument('--runs_name', type=str, default="rce_T_revision")
parser.add_argument('--project', type=str, default="rce_T_revision_partial")
parser.add_argument('--disable_mwn',
                    action='store_true')
parser.add_argument('--save',
                    action='store_true')
parser.add_argument('--analyze',
                    action='store_true')
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--print_predictions', action='store_true')
parser.add_argument('--temp_anneal', action='store_true')
parser.add_argument('--use_ground_truth', action='store_true')
parser.add_argument('--correct_only', action='store_true')
parser.add_argument('--outer_use_valid', action='store_true')
parser.add_argument('--ifplot', action='store_true')
parser.add_argument('--corruption_ratio', type=float, default=0.)
parser.add_argument('--T_init', type=float, default=4.5)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--meta_batch_size', type=int, default=1024)
parser.add_argument('--max_epoch', type=int, default=120)
parser.add_argument('--start_updating_T', type=int, default=20)
parser.add_argument('--start_correction', type=int, default=10)
parser.add_argument('--classes', type=int, nargs='+', default=None)
parser.add_argument('--decay_epoch', type=int, nargs='+', default=None)
parser.add_argument('--train_limits', type=int, default=None)
parser.add_argument('--meta_interval', type=int, default=1)
parser.add_argument('--paint_interval', type=int, default=20)


args = parser.parse_args()
print(args)
if args.wandb:
    os.system("wandb login --relogin 8bb1fef7b4815daa3cb2ec7c5b0b9ee40d7ea6ed")
    wandb.init(project=args.project, name=args.runs_name, config=args)

if args.decay_epoch is None:
    decay_epoch1, decay_epoch2 = args.max_epoch - 5, args.max_epoch - 2
else:
    decay_epoch1, decay_epoch2 = args.decay_epoch


def reweighting_revision_loss( out, target, T):
    loss = 0.
    eps = 1e-10
    out_softmax = F.softmax(out, dim=1)
    for i in range(len(target)):
        temp_softmax = out_softmax[i]
        temp = out[i]
        temp = torch.unsqueeze(temp, 0)
        temp_softmax = torch.unsqueeze(temp_softmax, 0)
        temp_target = target[i]
        temp_target = torch.unsqueeze(temp_target, 0)
        pro1 = temp_softmax[:, target[i]].detach()
        out_T = torch.matmul(T.t(), temp_softmax.t().detach())
        out_T = out_T.t()
        pro2 = out_T[:, target[i]]
        beta = pro1 / (pro2 + eps)
        cross_loss = F.cross_entropy(temp, temp_target)
        _loss = beta * cross_loss
        loss += _loss
    return loss / len(target)

def get_correct_percentage(T):
    diagonal = torch.diagonal(T, 0)
    column_sums = torch.sum(T, dim=0)
    fraction_correct = diagonal/column_sums
    return fraction_correct

def get_correct_data_idx(net, T, train_loader_unshuffled, fraction_correct):
    p_y_hat, labelslist = [], []
    for iteration, (inputs, labels) in enumerate(train_loader_unshuffled):
        net.eval()
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = net(inputs)
        prob = F.softmax(outputs, dim=1)
        out_forward = torch.matmul(prob ,T)
        p_y_hat.append(out_forward)
        labelslist.append(labels)
    idx_list = torch.tensor(list(range(len(train_loader_unshuffled.dataset)))).cuda()
    p_y_hat, labelslist = torch.cat(p_y_hat), torch.cat(labelslist)
    idx_correct = []
    for i in range(10):
        p_y_hat_i = p_y_hat[labelslist==i]
        idx_i = idx_list[labelslist==i]
        p_y_hat_i_correct, idx_i_correct = torch.topk(p_y_hat_i, int(fraction_correct[i]*len(p_y_hat_i)))
        idx_correct.append(idx_i[idx_i_correct])
    return idx_correct



def foward_loss( out, target, T):
    out_softmax = F.softmax(out, dim=1)
    p_T = torch.matmul(out_softmax , T)
    cross_loss = F.nll_loss(torch.log(p_T), target)
    return cross_loss, out_softmax, p_T

def error(T, T_true):
    error = np.sum(np.abs(T-T_true)) / np.sum(np.abs(T_true))
    return error

dim = 100 if args.dataset == "cifar100" else 10
if args.classes is not None:
    dim = len(args.classes)

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def DMI_loss(output, target):
    outputs = F.softmax(output, dim=1)
    targets = target.reshape(target.size(0), 1).cpu()
    y_onehot = torch.FloatTensor(target.size(0), 10).zero_()
    y_onehot.scatter_(1, targets, 1)
    y_onehot = y_onehot.transpose(0, 1).cuda()
    mat = y_onehot @ outputs
    return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)

def meta_weight_net():
    set_cudnn(device=args.device)
    set_seed(seed=args.seed)
    if args.dataset == "mnist":
        net = Lenet(dim).cuda()
    else:
        if args.network == "r18":
            net = ResNet18(dim).to(device=args.device).cuda()
        else:
            net = ResNet34(dim).to(device=args.device).cuda()
    reduction_points = [decay_epoch1, decay_epoch2]
    trans = sig_t("cuda:0", dim, init=args.T_init)
    trans = trans.cuda()
    t = trans()
    trans_matrix = t.detach().cpu().numpy()
    print(f"initial T {trans_matrix}")
    if args.meta_optim == "adam":
        meta_optimizer = torch.optim.Adam(trans.parameters(), lr=args.meta_lr)
    else:
        meta_optimizer = torch.optim.SGD(trans.parameters(), lr=args.meta_lr, weight_decay=0, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device=args.device)
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        dampening=args.dampening,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )
    scheduler1 = MultiStepLR(optimizer, milestones=reduction_points, gamma=0.1)
    scheduler2 = MultiStepLR(meta_optimizer, milestones=reduction_points, gamma=0.1)
    lr = args.lr

    train_dataloader, meta_laoder, test_dataloader, imbalanced_num_list, train_dataloader_unshuffled, corrupted_locations, correct_locations, _, _, true_targets, true_trans_matrix, train_dataloader_correct = build_dataloader(
        seed=args.seed,
        dataset=args.dataset,
        num_meta_total=args.num_meta,
        imbalanced_factor=args.imbalanced_factor,
        corruption_type=args.corruption_type,
        corruption_ratio=args.corruption_ratio,
        batch_size=args.batch_size,
        return_T=True,
        classes = args.classes,
        train_limits=args.train_limits
    )
    if args.correct_only:
        correct_dataset = train_dataloader.dataset
        correct_dataset.data = correct_dataset.data[correct_locations]
        correct_dataset.targets = list(np.array(correct_dataset.targets)[correct_locations])
        train_dataloader = DataLoader(correct_dataset, batch_size=100, shuffle=True, pin_memory=True, num_workers=5)
    if not args.outer_use_valid:
        meta_laoder = DataLoader(train_dataloader.dataset, batch_size=args.meta_batch_size, shuffle=True, pin_memory=True, num_workers=5)

    meta_dataloader_iter = iter(meta_laoder)
    num_layers = len([p for p in net.parameters()])
    print(f"{args.disable_mwn}")
    error_prev, net_best, T_est_best = 1000, None, None
    for epoch in range(args.max_epoch):

        print(f"length of meta loader: {len(meta_laoder.dataset)}, outer obj {args.outer_obj}")
        print('Training...')
        train_acc = 0.
        total_num = 0
        for iteration, (inputs, labels) in enumerate(train_dataloader):
            net.train()
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            if args.use_ground_truth:
                trans_matrix = torch.from_numpy(true_trans_matrix.astype(float)).float().cuda()
                trans_matrix = torch.full_like(trans_matrix, 1e-4)+trans_matrix
            else:
                trans_matrix = trans()
            if (iteration + 1) % args.meta_interval == 0 and not args.disable_mwn and (epoch+1)>= args.start_updating_T:
                if iteration % 50 == 0:
                    estimate_error = error(true_trans_matrix, (trans_matrix).cpu().detach().numpy())
                    print(f"error {estimate_error}")
                if args.dataset == "mnist":
                    pseudo_net = Lenet(dim).cuda()
                else:
                    if args.network == "r18":
                        pseudo_net = ResNet18(dim).to(device=args.device).cuda()
                    else:
                        pseudo_net = ResNet34(dim).to(device=args.device).cuda()
                time1 = time.time()
                pseudo_net.load_state_dict(net.state_dict())
                pseudo_net.train()
                old_params = [(n, p) for (n,p) in pseudo_net.named_parameters() if p.requires_grad]
                print(f"iteration {iteration}, len {len(old_params)}")
                #get output
                pseudo_outputs = pseudo_net(inputs)
                # trans_matrix = torch.autograd.Variable(trans_matrix, requires_grad=True)
                if args.loss == "reweight":
                    pseudo_loss = reweighting_revision_loss(pseudo_outputs, labels, trans_matrix)
                elif args.loss == "forward":
                    pseudo_loss, out_softmax, p_T = foward_loss(pseudo_outputs, labels, trans_matrix)
                # gather proxy inputs
                pseudo_grads = torch.autograd.grad(pseudo_loss, [p[1] for p in old_params], create_graph=True)

                pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=lr)
                pseudo_optimizer.load_state_dict(optimizer.state_dict())
                pseudo_optimizer.meta_step(old_params, pseudo_grads, if_update=True)
                time2 = time.time()
                print(f"meta time spent {time2 - time1}")

                del pseudo_grads

                try:
                    meta_inputs, meta_labels = next(meta_dataloader_iter)
                except StopIteration:
                    meta_dataloader_iter = iter(meta_laoder)
                    meta_inputs, meta_labels = next(meta_dataloader_iter)

                meta_inputs, meta_labels = meta_inputs.to(args.device), meta_labels.to(args.device)
                meta_outputs = pseudo_net(meta_inputs)
                if args.outer_obj == "rce":
                    one_hot = torch.zeros(len(meta_labels), dim).cuda().scatter_(1, meta_labels.view(-1, 1),  10).cuda()
                    one_hot = F.softmax(one_hot)
                    meta_loss_vector = F.softmax(meta_outputs, dim=1)*(F.log_softmax(meta_outputs, dim=1)-torch.log(one_hot)) - torch.mul(F.softmax(meta_outputs), F.log_softmax(meta_outputs))
                elif args.outer_obj == 'mae':
                    yhat_meta_1 = F.softmax(meta_outputs, dim=-1)
                    first_index = to_var(torch.arange(meta_outputs.size(0)).type(
                        torch.LongTensor), requires_grad=False)
                    yhat_meta_1 = yhat_meta_1[first_index, meta_labels]
                    meta_loss_vector = 2*torch.mean(1. - yhat_meta_1)
                elif args.outer_obj == 'gce':
                    loss = 0
                    for i in range(meta_outputs.size(0)):
                        loss += (1.0 - (meta_outputs[i][meta_labels[i]]) ** 1) / 1
                    meta_loss_vector = loss / meta_outputs.size(0)
                elif args.outer_obj == 'dmi':
                    meta_loss_vector = DMI_loss(meta_outputs, meta_labels)
                else:
                    raise NotImplementedError(f"{args.outer_obj} is not implemented")
                meta_loss = torch.mean(meta_loss_vector)
                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()

            outputs = net(inputs)
            if epoch >= args.start_correction:
                prob = F.softmax(outputs, dim=1)
                prob = prob.t()
                if args.loss == "reweight":
                    loss = reweighting_revision_loss(outputs, labels, trans_matrix.detach())
                elif args.loss == "forward":
                    loss, out_softmax, p_T = foward_loss(outputs, labels, trans_matrix.detach())
                out_forward = torch.matmul(trans_matrix.t(), prob)
                out_forward = out_forward.t()
            else:
                loss = F.cross_entropy(outputs, labels)
                out_forward = outputs
            pred = torch.max(out_forward, 1)[1]
            train_correct = (pred == labels).sum()
            train_acc += train_correct.item()
            total_num += len(labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler1.step()
        scheduler2.step()
        if args.print_predictions:
            with torch.no_grad():
                p_y = []
                p_y_hat = []
                labels_list = []
                for inputs, labels in train_dataloader_correct:
                    net.eval()
                    inputs, labels = inputs.to(args.device), labels.to(args.device)
                    outputs = net(inputs)
                    prob = F.softmax(outputs, dim=1)
                    out_forward = torch.matmul(prob ,trans_matrix)
                    p_y.append(prob)
                    p_y_hat.append(out_forward)
                    labels_list.append(labels)
                p_y = torch.cat(p_y, dim=0)
                p_y_hat = torch.cat(p_y_hat, dim=0)
                labels_list = torch.cat(labels_list, dim=0)
                for i in range(10):
                    print(f"py_{i}: {torch.mean(p_y[labels_list==i], dim=0)}")
                    print(f"py_hat_{i}: {torch.mean(p_y_hat[labels_list==i], dim=0)}")

        print(f"train loss {loss}")
        train_acc =  train_acc/ total_num
        estimate_error = error(true_trans_matrix, (trans_matrix).cpu().detach().numpy())
        if estimate_error < error_prev:
            net_best, T_est_best = copy.deepcopy(net), torch.clone(trans_matrix)
        print(f"trans_matrix {(trans_matrix).cpu().detach().numpy()}")
        print('Computing Test Result...')
        test_loss, test_accuracy = compute_loss_accuracy(
            net=net,
            data_loader=test_dataloader,
            criterion=criterion,
            device=args.device,
        )

        print('Epoch: {}, (Loss, Accuracy) Test: ({:.4f}, {:.2%}) LR: {}'.format(
            epoch,
            test_loss,
            test_accuracy,
            lr,
        ))
        log_dict = {
            "train_acc": train_acc,
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
        }
        if args.wandb:
            wandb.log(log_dict)
    if args.save:
        path = f"exps/{args.dataset}/{args.corruption_type}/{args.corruption_ratio}/files"
        if not os.path.exists(f"{path}"):
            os.makedirs(f"{path}")
        clean_pred = []
        noisy_pred = []
        targets_noisy = []
        with torch.no_grad():
            for iteration, (batch_x, batch_y) in enumerate(train_dataloader):
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                net_best.eval()
                clean = net_best(batch_x)
                clean = F.softmax(clean, dim=1)
                out = torch.mm(clean, T_est_best)
                clean_pred.append(clean)
                noisy_pred.append(out)
                targets_noisy.append(batch_y)
            clean_pred = torch.cat(clean_pred)
            noisy_pred = torch.cat(noisy_pred)
            targets_noisy = torch.cat(targets_noisy)
        torch.save({"true_T":true_trans_matrix, "est_T": T_est_best, "clean_pred": clean_pred, "noisy_pred": noisy_pred, "targets_noisy": targets_noisy}, f"{path}/files.pth")


if __name__ == '__main__':
    meta_weight_net()