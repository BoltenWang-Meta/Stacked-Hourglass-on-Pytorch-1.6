import torch
import time
from    torch import nn


class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResBlk, self).__init__()
        self.ch_in = ch_in
        ch_mid = int(ch_out / 2)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(ch_in),
            nn.ReLU(),
            nn.Conv2d(ch_in, ch_mid, kernel_size=1),
            nn.BatchNorm2d(ch_mid),
            nn.ReLU(),
            nn.Conv2d(ch_mid, ch_mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_mid),
            nn.ReLU(),
            nn.Conv2d(ch_mid, ch_out, kernel_size=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        )
    def forward(self, x):
        assert x.shape[1] == self.ch_in
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out = out1 + out2
        return out

class HourGlassBlk(nn.Module):
    def __init__(self):
        super(HourGlassBlk, self).__init__()
        self.stage_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            ResBlk(256, 256)
        )
        self.stage_1_b = ResBlk(256, 256)

        self.stage_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            ResBlk(256, 256)
        )
        self.stage_2_b = ResBlk(256, 256)

        self.stage_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            ResBlk(256, 256)
        )
        self.stage_3_b = ResBlk(256, 256)

        self.stage_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            ResBlk(256, 256),
            ResBlk(256, 256),
            ResBlk(256, 256),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        )
        self.stage_4_b = ResBlk(256, 256)

        self.state_3_u = nn.Sequential(
            ResBlk(256, 256),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        )

        self.state_2_u = nn.Sequential(
            ResBlk(256, 256),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        )

        self.state_1_u = nn.Sequential(
            ResBlk(256, 256),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        out1 = self.stage_1(x)
        out1_b = self.stage_1_b(x)

        out2 = self.stage_2(out1)
        out2_b = self.stage_2_b(out1)

        out3 = self.stage_3(out2)
        out3_b = self.stage_3_b(out2)

        out4 = self.stage_4(out3)
        out4_b = self.stage_4_b(out3)
        out4 = out4 + out4_b

        out3 = self.state_3_u(out4)
        out3 = out3 + out3_b

        out2 = self.state_2_u(out3)
        out2 = out2 + out2_b

        out1 = self.state_1_u(out2)
        out1 = out1 + out1_b
        return out1

class HGPoseNet(nn.Module):
    def __init__(self, landmark):
        super(HGPoseNet, self).__init__()
        self.conv_pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlk(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            ResBlk(128, 128),
            ResBlk(128, 256)
        )
        HG1 = HourGlassBlk()
        conv_1 = nn.Sequential(
            ResBlk(256, 256),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        conv_1_bottle = nn.Sequential(
            nn.Conv2d(256, landmark, kernel_size=1),
            nn.Conv2d(landmark, 256, kernel_size=1)
        )
        conv_1_add = nn.Conv2d(256, 256, kernel_size=1)
        HG_layer = [HG1, conv_1, conv_1_bottle, conv_1_add]
        self.HG_layers = []
        for i in range(7):
            self.HG_layers.append(HG_layer)


        self.end_stage = nn.Sequential(
            HourGlassBlk(),
            ResBlk(256, 256),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, landmark, kernel_size=1)
        )

    def forward(self, img):
        map_1 = self.conv_pre(img)
        for stage in self.HG_layers:
            map_1 = stage[0](map_1)
            map_1 = stage[1](map_1)
            map_1_1 = stage[2](map_1)
            map_1_2 = stage[3](map_1)
            map_1 = map_1_1 + map_1_2
        out = self.end_stage(map_1)
        return out

def test():
    model = HGPoseNet(16)
    x = torch.rand((1, 3, 256, 256))
    st = time.time()
    out1 = model(x)
    end = time.time()
    print(end-st)
    print(out1.shape)

if __name__ == '__main__':
    test()