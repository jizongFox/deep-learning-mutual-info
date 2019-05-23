import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from dataloader import get_data_loaders
from network import neural_network_model
from utils import get_information, plot_info_plane
# from calc_info import get_information
from deepclustering.utils import class2one_hot
from deepclustering.loss.loss import KL_div

device = torch.device('cuda')
net = neural_network_model()
net.cuda()
optimizer = optim.Adam(net.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[40,80,120],gamma=0.5)
criterion = KL_div()
train_loader, test_loader = get_data_loaders()

for i in range(1000):
    for _, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        preds, features = net(data)
        loss = criterion(preds, class2one_hot(labels, 2).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        info = get_information([[w.detach().cpu().numpy() for w in features]],
                               data.cpu().numpy(),
                               class2one_hot(labels, 2).cpu().numpy(),
                               epoch_num=0)
        acc = ((preds.max(1)[1] == labels.long()).sum().float() / labels.shape[0]).item()
    print(f"loss: {loss.item():.4f}, acc:{acc:.4f}")
    plot_info_plane(i, [_['local_IXT'] for _ in info], [_['local_ITY'] for _ in info])
    scheduler.step()
