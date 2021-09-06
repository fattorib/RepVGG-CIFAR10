import argparse
from copy import Error
import os
import shutil
import time

import torch

# from torch._C import R
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP

import math
import numpy as np


from RandAugment import RandAugment
from torchvision.datasets import CIFAR10, ImageFolder, CIFAR100
import wandb

from RepVGG import (
    RepVGG,
    create_RepVGG_B0,
    deploy_model,
    create_RepVGG_A0,
    create_RepVGG_A1,
    create_RepVGG_A2,
    create_RepVGG_B0,
    create_RepVGG_B1,
    create_RepVGG_B2,
    create_RepVGG_B3,
)
from ResNet import ResNet
from utils.utils import mixup_data, mixup_criterion, LabelSmoothingCrossEntropy


try:
    import torch.cuda.amp as amp
except ImportError:
    raise ImportError("Your version of PyTorch is too old.")

best_prec1 = 0


def parse():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    parser.add_argument(
        "-data",
        "--data",
        default="ML/",
        type=str,
        metavar="DIR",
        help="path to dataset",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )

    parser.add_argument(
        "--epochs",
        default=120,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=128,
        type=int,
        metavar="N",
        help="mini-batch size per process (default: 128)",
    )

    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )

    parser.add_argument(
        "--print-freq",
        "-p",
        default=100,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )

    parser.add_argument("--local_rank", default=0, type=int)

    # My additional args
    parser.add_argument("--model", type=str, default="RepVGG")
    parser.add_argument("--CIFAR10", type=bool, default=False)
    parser.add_argument("--CIFAR100", type=bool, default=False)
    parser.add_argument("--Imagenet200", type=bool, default=False)
    parser.add_argument("--RandAugN", type=int, default=1)
    parser.add_argument("--RandAugM", type=int, default=2)
    parser.add_argument("--Mixed-Precision", type=bool, default=True)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--cos-anneal", type=bool, default=False)
    parser.add_argument("--step-lr", type=bool, default=False)
    parser.add_argument("--disable-cos", type=bool, default=False)
    parser.add_argument("--base-lr", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=bool, default=False)

    args = parser.parse_args()
    return args


def main():
    global best_prec1, args

    args = parse()

    cudnn.benchmark = True

    if torch.cuda.is_available():

        if args.model == "RepVGGA0":
            model = create_RepVGG_A0(num_classes=args.num_classes)

        if args.model == "RepVGGA1":
            model = create_RepVGG_A1(num_classes=args.num_classes)

        if args.model == "RepVGGA2":
            model = create_RepVGG_A2(num_classes=args.num_classes)

        if args.model == "RepVGGB0":
            model = create_RepVGG_B0(num_classes=args.num_classes)

        if args.model == "RepVGGB1":
            model = create_RepVGG_B1(num_classes=args.num_classes)

        if args.model == "RepVGGB2":
            model = create_RepVGG_B2(num_classes=args.num_classes)

        if args.model == "RepVGGB3":
            model = create_RepVGG_B2(num_classes=args.num_classes)

        elif args.model == "ResNet":
            model = ResNet(filters_list=[16, 32, 64], N=3)

        else:
            pass

        model = model.cuda()

        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropy().cuda()
        else:
            criterion = nn.CrossEntropyLoss().cuda()

    if args.cos_anneal:
        assert args.step_lr == False
        # optimizer = torch.optim.SGD(
        #     model.parameters(),
        #     lr=args.base_lr,
        #     weight_decay=args.weight_decay,
        #     momentum=0.9,
        #     nesterov=True,
        # )

        optimizer = create_optimizer(model, args.weight_decay, args.base_lr)

    if args.step_lr:
        assert args.cos_anneal == False

        if args.adam:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-2,
                weight_decay=args.weight_decay,
            )
        else:

            optimizer = create_optimizer(model, args.weight_decay, args.base_lr)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[130, 180], gamma=0.1
        )

    if args.CIFAR10:
        assert args.num_classes == 10, "Must have 10 output classes for CIFAR10"
        # Use CIFAR-10 data augmentations
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(
                    (32, 32),
                    padding=4,
                    fill=0,
                    padding_mode="constant",
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_train.transforms.insert(0, RandAugment(args.RandAugN, args.RandAugM))

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        train_dataset = CIFAR10(
            root="./CIFAR", train=True, download=True, transform=transform_train
        )

        train_dataset, validation_dataset = torch.utils.data.random_split(
            train_dataset, [45000, 5000]
        )

        test_dataset = CIFAR10(
            root="./CIFAR", train=False, download=True, transform=transform_test
        )

    if args.CIFAR100:
        assert args.num_classes == 100, "Must have 100 output classes for CIFAR100"
        # Use CIFAR-100 data augmentations
        transform_train = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.RandomCrop(
                    (32, 32), padding=4, fill=0, padding_mode="constant"
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                ),
            ]
        )
        transform_train.transforms.insert(0, RandAugment(args.RandAugN, args.RandAugM))

        transform_test = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4913, 0.4821, 0.4465], std=[0.2470, 0.2434, 0.2615]
                ),
            ]
        )

        train_dataset = CIFAR100(
            root="./CIFAR100", train=True, download=True, transform=transform_train
        )

        train_dataset, validation_dataset = torch.utils.data.random_split(
            train_dataset, [45000, 5000]
        )

        test_dataset = CIFAR100(
            root="./CIFAR100", train=False, download=True, transform=transform_test
        )

    if args.Imagenet200:
        assert args.num_classes == 200, "Must have 200 output classes for Imagenet200"
        transform_train = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.RandomCrop(
                    (64, 64), padding=8, fill=0, padding_mode="constant"
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.480, 0.448, 0.397], std=[0.272, 0.265, 0.274]
                ),
            ]
        )
        transform_train.transforms.insert(0, RandAugment(args.RandAugN, args.RandAugM))

        transform_test = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.480, 0.448, 0.397], std=[0.272, 0.265, 0.274]
                ),
            ]
        )
        train_dataset = ImageFolder(
            root="./tiny-imagenet-200/train", transform=transform_train
        )

        train_dataset, validation_dataset = torch.utils.data.random_split(
            train_dataset, [85000, 15000]
        )
        test_dataset = ImageFolder(
            root="./tiny-imagenet-200/val", transform=transform_train
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Setup WandB logging here

    wandb_run = wandb.init(project="RepVGG")
    wandb.config.max_epochs = args.epochs
    wandb.config.batch_size = args.batch_size
    wandb.config.weight_decay = args.weight_decay
    wandb.config.RandAugmentN = args.RandAugN
    wandb.config.RandAugmentM = args.RandAugM

    wandb.config.ModelName = args.model
    wandb.config.Dataset = "CIFAR10" if args.CIFAR10 else "TinyImagenet"

    scaler = None
    if args.Mixed_Precision:
        scaler = amp.GradScaler()

    global mixup
    mixup = False
    if args.mixup > 0:
        mixup = True

    for epoch in range(0, args.epochs):

        if args.cos_anneal:
            lr = adjust_learning_rate(optimizer, epoch, args)
            scheduler = None

        train_loss = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scaler=scaler,
            scheduler=scheduler,
        )
        if args.step_lr:
            scheduler.step()
            lr = (scheduler.get_last_lr())[0]

        _, _, val_loss = validate(validation_loader, model, criterion)

        if epoch % 10 == 0:

            prec1, prec5, test_loss = validate(test_loader, model, criterion)
            wandb.log(
                {
                    "acc@1": prec1,
                    "acc@5": prec5,
                    "Learning Rate": lr,
                    "Training Loss": train_loss,
                    "Validation Loss": val_loss,
                }
            )

        else:
            wandb.log(
                {
                    "Learning Rate": lr,
                    "Training Loss": train_loss,
                    "Validation Loss": val_loss,
                }
            )

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
            )


def train(train_loader, model, criterion, optimizer, epoch, scaler, scheduler=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):

        optimizer.zero_grad()

        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()

        if scaler is not None:
            with amp.autocast():
                if mixup:
                    images, targets_a, targets_b, lam = mixup_data(
                        images, target, args.mixup
                    )
                    output = model(images)
                    loss = mixup_criterion(output, targets_a, targets_b, lam)
                else:
                    output = model(images)
                    loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Measure accuracy
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        reduced_loss = loss.item()

        # to_python_float incurs a host<->device sync
        losses.update((reduced_loss), images.size(0))
        top1.update((prec1), images.size(0))
        top5.update((prec5), images.size(0))

        if i % args.print_freq == 0 and args.local_rank == 0:
            print(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t Loss {losses.test:.10f} ({losses.avg:.4f})\t Prec@1 {top1.test.item():.3f} ({top1.avg.item():.3f})Prec@5 {top5.test.item():.3f} ({top5.avg.item():.3f})"
            )

    return losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.test = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, test, n=1):
        self.test = test
        self.sum += test * n
        self.count += n
        self.avg = self.sum / self.count


def validate(loader, model, criterion, scaler=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # Update the reparam
    if args.model == "RepVGG":
        deployed_model = deploy_model(model)

    elif args.model == "RepVGGRef":
        deployed_model = repvgg_model_convert(model)

    end = time.time()

    for i, (images, target) in enumerate(loader):
        # compute output
        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()
        with torch.no_grad():
            if args.model == "RepVGG":
                output = deployed_model(images)
                loss = criterion(output, target)

            elif args.model == "RepVGGRef":
                output = deployed_model(images)
                loss = criterion(output, target)

            else:
                output = model(images)
                loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        reduced_loss = loss.item()

        losses.update((reduced_loss), images.size(0))
        top1.update((prec1), images.size(0))
        top5.update((prec5), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(f"* Prec@1 {top1.avg.item():.3f} Prec@5 {top5.avg.item():.3f}")

    return top1.avg.item(), top5.avg.item(), losses.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified testues of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.base_lr
    if hasattr(args, "warmup") and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    else:
        lr *= 0.5 * (
            1.0
            + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup))
        )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    # for tracking
    return lr


def create_optimizer(model, weight_decay, lr):
    params = []
    for key, value in model.named_parameters():

        if (
            "fc.bias" in key
            or "bias" in key
            or "bn" in key
            # or "conv_3" in key
            # or "conv_1" in key
        ):
            print(f"No weight decay for paramater: {key}")
            apply_weight_decay = 0
            params += [
                {"params": [value], "lr": lr, "weight_decay": apply_weight_decay}
            ]

        else:
            apply_weight_decay = weight_decay
            params += [
                {"params": [value], "lr": lr, "weight_decay": apply_weight_decay}
            ]

    return torch.optim.SGD(params, lr, momentum=0.9, nesterov=True)


if __name__ == "__main__":
    main()
