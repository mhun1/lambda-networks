import torch
from model import Lambda_Unet
from skimage import io, transform
from dataset import FluoDataset
from torchvision import transforms, utils
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from torch import optim
import torch.nn as nn

def train(net,dataset,epochs=10,batch_size=10,lr=0.99,device="cpu"):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)

    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epochs in range(epochs):
        net.train()
        epoch_loss = 0
        for batch in train_loader:
            x = batch["image"].to(device=device)
            y = batch["ground_truth"].to(device=device)
            
            pred = net(x)
            loss = criterion(pred,y)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




if __name__ == '__main__':
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine((0, 360)),
        transforms.Resize(512),
        transforms.ToTensor(),

    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tr_dataset = FluoDataset("/home/mhun/data/Fluo-N2DH-GOWT1/01/", "/home/mhun/data/Fluo-N2DH-GOWT1/01_GT/TRA/",
                             transform=data_transform, target_transform=data_transform)

    net = Lambda_Unet(n_classes=1)

    net.to(device=device)

    train(net,tr_dataset,epochs=10,batch_size=1,lr=0.99,device=device)