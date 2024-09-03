import sys




import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from subnet import subnet
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as tfs
from advertorch.attacks import LinfPGDAttack

from dist_kd import DIST
dist_kd = DIST(1.0,2.0,2.0)

bs = 32
img_size=32
image_gap=3
CIFAR100_path = "./cifar100"
train_transforms =  tfs.Compose([ tfs.ToTensor(),tfs.RandomOrder([
        tfs.RandomApply(
            [
                tfs.Resize((img_size + image_gap, img_size + image_gap)),
                tfs.CenterCrop((img_size, img_size)),
            ],0.5
        ),
        tfs.RandomHorizontalFlip(),
        tfs.RandomApply(
            [tfs.GaussianBlur(3, sigma=(0.1, 1.0))],0.5
        ),
        tfs.RandomApply(
            [tfs.RandomErasing(scale=(0.02, 0.22))],0.5
        ),
    ]),
        tfs.Resize(32)]
        )
transforms = tfs.Compose(
            [tfs.Resize(32), tfs.ToTensor(),]
        )

def acc_estimate(net,data_loader):
    # ----------
    #  estimate
    # ----------
    with torch.no_grad():
        correct_netD = 0.0
        total = 0.0
        for inputs, labels in data_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_netD += (predicted == labels).sum()
            if correct_netD > 200:
                break
    print('Accuracy of the ori-network: %.2f %%' %
              (100. * correct_netD.float() / total))
    return (100. * correct_netD.float() / total)

def get_att_results(submodel,targetmodel, data_loader,target):
    adversary = LinfPGDAttack(
                submodel,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=8.0 / 255, eps_iter=2.0/255,
                clip_min=0.0, clip_max=1.0,
                targeted=target)
    correct = 0.0
    total = 0.0
    total_L2_distance = 0.0
    att_num = 0.
    acc_num = 0.
    for data in data_loader:
        inputs, labels = data
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")
        with torch.no_grad():
            outputs = targetmodel(inputs)
            _, predicted = torch.max(outputs.data, 1)
        if target:
            # randomly choose the specific label of targeted attack
            labels = torch.randint(0, 100 - 1, (inputs.size(0),)).cuda()
            # test the images which are not classified as the specific label
            ones = torch.ones_like(predicted)
            zeros = torch.zeros_like(predicted)
            acc_sign = torch.where(predicted == labels, zeros, ones)
            acc_num += acc_sign.sum().float()
            adv_inputs_ori = adversary.perturb(inputs, labels)
            L2_distance = (adv_inputs_ori - inputs).squeeze()
            L2_distance = (torch.linalg.norm(L2_distance.flatten(start_dim=1), dim=1)).data
            L2_distance = L2_distance * acc_sign
            total_L2_distance += L2_distance.sum()
            with torch.no_grad():
                outputs = targetmodel(adv_inputs_ori)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum()
                att_sign = torch.where(predicted == labels, ones, zeros)
                att_sign = att_sign + acc_sign
                att_sign = torch.where(att_sign == 2, ones, zeros)
                att_num += att_sign.sum().float()
        else:
            ones = torch.ones_like(predicted)
            zeros = torch.zeros_like(predicted)
            acc_sign = torch.where(predicted == labels, ones, zeros)
            acc_num += acc_sign.sum().float()

            adv_inputs_ori = adversary.perturb(inputs, labels)
            L2_distance = (adv_inputs_ori - inputs).squeeze()
            L2_distance = (torch.linalg.norm(L2_distance.flatten(start_dim=1), dim=1)).data
            L2_distance = L2_distance * acc_sign
            total_L2_distance += L2_distance.sum()
            with torch.no_grad():
                outputs = targetmodel(adv_inputs_ori)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum()
                att_sign = torch.where(predicted == labels, zeros, ones)
                att_sign = att_sign + acc_sign
                att_sign = torch.where(att_sign == 2, ones, zeros)
                att_num += att_sign.sum().float()
        if acc_num > 1000:
            break
    att_result = (att_num / acc_num * 100.0)
    print('Attack success rate: %.2f %%--l2:  %.4f ' % (att_result,total_L2_distance / acc_num))
    return att_result

# PES generated code
def Gen_PES(sub_net, inputs,labels_tar,eps):
    inputs.requires_grad = True
    output = sub_net(inputs)
    loss = F.cross_entropy(output, labels_tar)
    sub_net.zero_grad()
    loss.backward()
    data_grad = inputs.grad.data
    sign_data_grad = data_grad
    perturbed_image = inputs - eps*sign_data_grad
    return perturbed_image

def get_target_label(random_numbers):
    # random
    other_numbers = []
    for num in random_numbers:
        while True:
            other_num = random.randint(0, 99)
            if other_num != num:
                other_numbers.append(other_num)
                break
    return torch.tensor(other_numbers).cuda()

def trainer(black_net,sub_net,train_loader,test_loader,estimate_loader):
    acc_estimate(black_net, estimate_loader)
    acc_estimate(black_net, test_loader)

    optimizer_D = torch.optim.SGD(sub_net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    query_num = 0
    print_epoch=0
    print_time=100

    while True:
        for data in train_loader:
            sub_net.train()
            sub_net.zero_grad()
            inputs, labels = data
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")

            sub_net.eval()
            # PES
            labels_tar = get_target_label(labels)
            eps = abs(np.random.normal(0, 3))
            perturbed_data = Gen_PES(sub_net, inputs,labels_tar,eps)

            sub_net.train()
            sub_net.zero_grad()
            output1 = sub_net(perturbed_data.detach())
            adv_pred = F.softmax(output1, dim=1)
            with torch.no_grad():
                outputs2 = F.softmax(black_net(perturbed_data.detach()), dim=1)
                _, label2 = torch.max(outputs2.data, 1)
                query_num += perturbed_data.size(0)
                print_epoch += 1
            # Lrp
            LRP_CE = F.cross_entropy(output1, label2)
            LRP = dist_kd(adv_pred, outputs2)
            # Lkd
            with torch.no_grad():
                outputs1 = F.softmax(black_net(inputs), dim=1)
                _, label1 = torch.max(outputs1.data, 1)
                query_num += inputs.size(0)
                print_epoch += 1
            fake_class_outputs = sub_net(inputs)
            fake_pred_label = F.softmax(fake_class_outputs, dim=1)
            LKD_CE = F.cross_entropy(fake_class_outputs, label1)
            LKD = dist_kd(fake_pred_label, outputs1)
            alpha = 10
            err_Dall = LKD + LKD_CE + alpha * (LRP + LRP_CE)  # - adv_loss
            err_Dall.backward()
            optimizer_D.step()
            # # estimate
            sub_net.eval()
            if print_epoch % print_time == 0:
                # Acc of substitute model
                acc_result = acc_estimate(sub_net,test_loader)
                print('ACC:' + str(acc_result.item()))
                print('Query num:' + str(query_num))
                # Target attack success rate
                target_result = get_att_results(sub_net, black_net, test_loader,target=True)
                print('Tar ASR:' + str(target_result.item()))
                # Non-Target attack success rate
                att_result = get_att_results(sub_net, black_net, test_loader,target=False)
                print('Non-tar ASR:' + str(att_result.item()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targetnet",
        type=str,
        default='res50-100'
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default='./resnet50.pth'
    )
    parser.add_argument(
        "--Datafree",
        type=str,
        default='fulldata'
    )
    opt = parser.parse_args()

    # init target model
    if opt.targetnet=='res50-100':
        from models.resnet import resnet50
        black_net = resnet50(100).cuda()
        state_dict = torch.load(opt.target_path)
    black_net.load_state_dict(state_dict)
    black_net = nn.DataParallel(black_net)
    black_net.eval()

    # init substitute model
    sub_net = subnet(100).cuda()
    if opt.Datafree=='fulldata':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=CIFAR100_path, train=True,
                              download=True,
                              transform=train_transforms
                              ),
            batch_size=bs,
            shuffle=True, num_workers=8
        )

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=CIFAR100_path, train=False,
                          download=True,
                          transform=transforms
                          ),
        batch_size=bs,
        shuffle=True,
        num_workers=8
    )
    estimate_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root=CIFAR100_path, train=True,
                          download=True,
                          transform=transforms
                          ),
        batch_size=bs,
        shuffle=True,
        num_workers=8
    )

    trainer(black_net,sub_net,train_loader,test_loader,estimate_loader)
