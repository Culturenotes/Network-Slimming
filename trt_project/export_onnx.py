from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import models
from torchvision import datasets, transforms

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--using_pruned_model', '-pruned', dest='pruned', action='store_true',
                    help='refine pruned model')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--arch', default='resnet', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=20, type=int,
                    help='depth of the neural network')
parser.add_argument('--save', default=r'C:/Users/liujiwei/Desktop/Network-Slimming/trt_project/onnx/',
                    help='the onnx model save path')
parser.add_argument('--saved', default=r'C:/Users/liujiwei/Desktop/Network-Slimming/logs/',
                    help='the model saved of the path ')
parser.add_argument('--name', default='',
                    help='the name of the model')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()


def test(model):
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(r'C:/Users/liujiwei/Desktop/Network-Slimming/data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    with torch.no_grad():
        correct = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = data,target
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

def read_cfg(savepath):
    with open(savepath, "r") as fp:
        data = [x.split()[0][1] if x =="'M'" else int(x) for x in fp.readlines()[1][1:-2].replace(",", "").split()]
    return data

def network():
    if args.pruned:
        datanames = os.listdir(args.saved)
        for dataname in datanames:
            if os.path.splitext(dataname)[1] == '.txt':  # 目录下包含.txt的文件
                print(dataname)
                cfg = read_cfg(args.saved + dataname)
                print(cfg)
        model = models.__dict__[args.arch](depth=args.depth, dataset=args.dataset, cfg=cfg).cuda()
    else:
        model = models.__dict__[args.arch](depth=args.depth, dataset=args.dataset).cuda()

    state_dict = torch.load(args.saved + args.name + ".pth", map_location=lambda storage, loc: storage)["state_dict"]
    model.load_state_dict(state_dict)
    onnx(model,args.save+args.name)
    test(model)
    params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Number of Parameters: %.1fM" % (params / 1e6))

def onnx(model,name=""):
    dummy_input = torch.ones(1, 3, 32, 32, dtype=torch.float32).cuda()
    torch.onnx.export(model,   # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      name+'.onnx',
                      verbose=True,) # store the trained parameter weights inside the model file

if __name__ =="__main__":
    network()