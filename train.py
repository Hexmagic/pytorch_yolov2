from operator import mod
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset.dataset import VOCDataset, detection_collate
from model.loss import Loss
from model.yolov2 import YOLOv2
from argparse import ArgumentParser


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    par = ArgumentParser()
    par.add_argument('--batch_size', type=int, default=16)
    arg = par.parse_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    dataset = VOCDataset('VOCdevkit', split='trainval')
    dataloader = DataLoader(
        dataset,
        batch_size=arg.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=detection_collate
    )
    dataset[0]
    torch.backends.cuda.enabled = True
    torch.backends.cuda.benchmark = True
    model = YOLOv2().to(device)
    criterion = Loss().cuda()
    lr = 1e-4
    opt = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(160):
        i = 0
        if epoch == 60:
            adjust_learning_rate(opt, lr/10)
        if epoch == 90:
            adjust_learning_rate(opt, lr/100)
        for batch in dataloader:
            i += 1
            img, boxes, label, num_obj = batch
            img = Variable(img).to(device)
            boxes = Variable(boxes).to(device)
            output = model(img)
            target = boxes, label, num_obj
            box_loss, iou_loss, class_loss = criterion(output, target)
            opt.zero_grad()
            loss = box_loss.mean() + iou_loss.mean() \
                + class_loss.mean()
            if i % 10 == 0:
                print(
                    f"batch {epoch} {i}/{len(dataloader)} loss:{round(loss.item(),3)} box: {round(box_loss.mean().item(),3)} iou: {round(iou_loss.mean().item(),3)} class: {round(class_loss.mean().item(),3)}")
            loss.backward()
            opt.step()
        if epoch % 5 == 0:
            print(f"epoch {epoch} save model")
            torch.save(model, f'weights/yolov2_{epoch}.pth')


if __name__ == '__main__':
    main()
