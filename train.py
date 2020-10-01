from operator import mod
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data.dataset import VOCDataset
from yolov2 import YOLOv2
def main():
    if torch.cuda.is_avaliable():
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
    torch.backends.cuda.enabled=True
    torch.backends.cuda.benchmark=True
    model = YOLOv2().to(device)
    opt = SGD(model.parameters(),lr=1e-4,momentum=0.9,weight_decay=5e-4)
    for epoch in range(160):
        for batch in dataloader:
            img,boxes,label,num_obj = batch
            img = Variable(img).to(device)
            ouput = model(img)
            import pdb; pdb.set_trace()
if __name__=='__main__':
    main()