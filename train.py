from operator import mod
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data.dataset import VOCDataset
from loss import Loss
from yolov2 import YOLOv2
def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dataset = VOCDataset('VOCdevkit',split='train')
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8        
    )
    dataset[0]
    torch.backends.cuda.enabled=True
    torch.backends.cuda.benchmark=True
    model = YOLOv2().to(device)
    criterion = Loss().cuda()
    opt = SGD(model.parameters(),lr=1e-4,momentum=0.9,weight_decay=5e-4)
    for epoch in range(160):
        for batch in dataloader:
            img,boxes,label,num_obj = batch
            img = Variable(img).to(device)
            ouput = model(img)
            target = (boxes,label,num_obj)
            opt.zero_grad()
            loss = criterion(ouput,target)
            loss.backward()
            opt.step()
if __name__=='__main__':
    main()