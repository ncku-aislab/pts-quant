import os
import time
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def build_imagenet_data(data_path: str = '', input_size: int = 224, batch_size: int = 64, workers: int = 4):
    print('==> Using Pytorch Dataset')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    traindir = os.path.join(data_path, 'ILSVRC2012_img_train')
    valdir = os.path.join(data_path, 'val')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    return train_loader, val_loader

def get_train_samples(train_loader, num_samples):
    train_data, target = [], []
    for batch in train_loader:
        train_data.append(batch[0])
        target.append(batch[1])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples], torch.cat(target, dim=0)[:num_samples]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    Returns a dict: {"Acc@1": ..., "Acc@5": ..., ...}
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
        pred = pred.t()  # [maxk, B]
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # [maxk, B]

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[f"Acc@{k}"] = correct_k.mul_(100.0 / batch_size).item()
        return res


@torch.no_grad()
def validate_model(val_loader, model, device=None, print_freq=100, topk=(1, 5)):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)

    print(device)
    batch_time = AverageMeter('Time', ':6.3f')
    acc_meters = {f"Acc@{k}": AverageMeter(f"Acc@{k}", ':6.2f') for k in topk}
    progress = ProgressMeter(
        len(val_loader),
        [batch_time] + list(acc_meters.values()),
        prefix='Test: ')

    model.eval()
    end = time.time()

    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        output = model(images)

        acc_dict = accuracy(output, target, topk=topk)
        for k in topk:
            acc_meters[f"Acc@{k}"].update(acc_dict[f"Acc@{k}"], images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    # Final print
    acc_summary = ' '.join([f"{k} {v.avg:.2f}" for k, v in acc_meters.items()])
    print(f' * {acc_summary}')

    # Return all top-k accuracies
    return {k: meter.avg for k, meter in acc_meters.items()}
