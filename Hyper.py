import torch
from    torch import nn
from    model.model import HGPoseNet
from    DataSet.LSP import LSPSet
from    torch.utils.data import DataLoader

def Hyper(path):
    net = HGPoseNet(15)
    hyper = {'lr': 1e-4, 'bz': 32, 'dv': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
             'ep': 2}
    LossFn = nn.MSELoss().to(hyper['dv'])
    optimier = torch.optim.SGD(net.parameters(), lr=hyper['lr'], momentum=0.7)

    train_set = LSPSet(path, img_size=256, HG_size=64, mode='train')
    test_set = LSPSet(path, img_size=256, HG_size=64, mode='test')

    train_loader = DataLoader(train_set, batch_size=hyper['bz'],
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=hyper['bz'],
                             shuffle=False, num_workers=2)
    return net, hyper, LossFn, optimier, train_loader, test_loader