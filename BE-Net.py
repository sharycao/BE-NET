from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import math

class spectralgradconv(nn.Module):
    """
    光谱梯度感知
    1.需要初始化权重形状
    2.使用拉普拉斯算子
    """
    def __init__(self, inch, outch, groups, ks=5, stride=1, first_layer='False'):
        super(spectralgradconv, self).__init__()
        self.first = first_layer
        self.padding = int(ks // 2)
        self.stride = (stride,stride)
        self.inch, self.outch, self.groups = int(inch), int(outch), int(groups)
        self.weight = nn.Parameter(torch.Tensor(self.outch, self.inch // groups, ks, ks))
        self.conv3x3 = nn.Conv2d(self.outch, self.outch, 3, 1, 1)
        self.bn = nn.BatchNorm2d(self.outch)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()
        self.gmaxpool = torch.nn.AdaptiveMaxPool2d(1)
        self.gavgpool = torch.nn.AdaptiveAvgPool2d(1)


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


    def forward(self,x):
        epsilon = 1e-10
        list_x = []
        for i in range(self.weight.shape[1]):  # i取决于conv的输入通道数
            list_x.append(torch.tensor([[-2, -4,-4,-4, -2], [-4, 0, 8, 0, -4], [-4, 8, 24, 8, -4], [-4, 0, 8, 0, -4],[-2, -4,-4,-4, -2]], device=x.device))
        list_x = torch.stack(list_x, 0)
        weight_x = torch.mul(self.weight, list_x)
        input_x = F.conv2d(x, weight_x, stride=self.stride, padding=self.padding,  groups=self.groups)
        conv = self.relu(self.bn(self.conv3x3(input_x)))

        #熵权
        gmax_group1, gavg_group1 = self.gmaxpool(conv), self.gavgpool(conv)
        cat = torch.cat([gmax_group1.squeeze(-1),gavg_group1.squeeze(-1)],dim=2)
        B, C, N = cat.shape

        probs = torch.softmax(cat, dim=2)
        log_probs = torch.log(probs + epsilon)  # 防止出现零
        entropy = -torch.sum(probs * log_probs, dim=2)
        max_entropy = torch.log(torch.tensor(N, dtype=torch.float32))  # 用于归一化熵值
        normalized_entropy = entropy / max_entropy

        weights = 1 - normalized_entropy  # 信息熵越大，权重越低

        out = weights.unsqueeze(-1).unsqueeze(-1)* conv + conv

        return out

class ChannelBoundaryAttention(nn.Module):
    def __init__(self, inchannel):
        """
                输入包括concatenate之后的skip+decoder,和decoder
                整体思路，解码器特征分两部分，一部分与编码器浅层特征concatenate，而后进行多尺度分组卷积；另一部分通过1x1conv调整channel大小，
                而后使用全局最大池化和全局平均池化进行通道注意力权重分配
                与2相比，减少了与decoder layer的共性特征融合
        """
        super(ChannelBoundaryAttention, self).__init__()
        self.groupchannel = inchannel // 4
        self.groupconv1 = nn.Conv2d(in_channels=self.groupchannel, out_channels= inchannel, kernel_size=1,padding=0)
        self.groupconv3 = nn.Conv2d(in_channels=self.groupchannel, out_channels= inchannel, kernel_size=3,padding=1)
        self.groupconv5 = nn.Conv2d(in_channels=self.groupchannel, out_channels= inchannel, kernel_size=3,dilation=2,
                                 padding=2)
        self.groupconv7 = nn.Conv2d(in_channels=self.groupchannel, out_channels= inchannel, kernel_size=3,
                                 dilation=3, padding=3)

        self.gmaxpool = torch.nn.AdaptiveMaxPool2d(1)
        self.gavgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.epsilon = 1e-10
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, skip, decoderx):
        skip_decoder_cat = torch.cat([skip, decoderx], dim=1)

        group1 = self.groupconv5(skip_decoder_cat[:, :self.groupchannel, :, :])
        group3 = self.groupconv7(skip_decoder_cat[:, self.groupchannel * 3:, :, :])
        group5 = self.groupconv1(skip_decoder_cat[:, int(self.groupchannel * 2):int(self.groupchannel * 3), :, :])
        group7 = self.groupconv3(skip_decoder_cat[:, int(self.groupchannel):int(self.groupchannel * 2), :, :])
        groupconv = group1 + group3 + group5 + group7
        groupconv_ = self.relu(groupconv)
        gmax_group1, gavg_group1 = self.gmaxpool(groupconv_), self.gavgpool(groupconv_)

        cat = torch.cat([gmax_group1.squeeze(-1), gavg_group1.squeeze(-1)], dim=2)
        B, C, N = cat.shape

        probs = torch.softmax(cat, dim=2)
        log_probs = torch.log(probs + self.epsilon)  # 防止出现零
        entropy = -torch.sum(probs * log_probs, dim=2)
        max_entropy = torch.log(torch.tensor(N, dtype=torch.float32))  # 用于归一化熵值
        normalized_entropy = entropy / max_entropy

        weights = 1 - normalized_entropy  # 信息熵越大，权重越低

        out = weights.unsqueeze(-1).unsqueeze(-1) * groupconv_ + groupconv_

        # dd = gmax_group1 - gavg_group1
        # dd = dd * groupconv
        # dd = self.tanh(dd) + 1
        # out = dd + groupconv
        return out

class pyrimad_decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(pyrimad_decoder,self).__init__()

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel)
                                     )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3,dilation=2,padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel)
                                     )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, dilation=3, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel)
        )
        # self.NLA = NLA(out_channel)
        self.maxpool3x3 = nn.MaxPool2d(kernel_size=3,padding=1,stride=1)
        self.maxpool5x5 = nn.MaxPool2d(kernel_size=5, padding=2, stride=1)
        self.maxpool7x7 = nn.MaxPool2d(kernel_size=7, padding=3, stride=1)
        self.sigmoid = nn.Sigmoid()

        self.epsilon = 1e-10
    def forward(self,x):
        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x1)
        x3 = self.conv7x7(x2)

        out = x1 + x2 + x3
        B, C, H, W = out.shape
        # 边界
        probs = torch.softmax(out, dim=1)
        log_probs = torch.log(probs + self.epsilon)
        entropy = -torch.sum(probs * log_probs, dim=1)
        max_entropy = torch.log(torch.tensor(C, dtype=torch.float32))#用于归一化熵值
        normalized_entropy = entropy / max_entropy

        # weights = 1 - normalized_entropy#信息熵越大，权重越低


        # 整体
        area_ = self.sigmoid(out)
        a3x3 = self.maxpool3x3(area_)
        a5x5 = self.maxpool5x5(area_)
        a7x7 = self.maxpool7x7(area_)
        area = (a3x3+a5x5+a7x7)/3 * area_

        output = out * (normalized_entropy.unsqueeze(1)+area) + out


        return output

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        # self.Up_conv5 = conv_block(filters[4], filters[3])
        self.Up_conv5 = pyrimad_decoder(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        # self.Up_conv4 = conv_block(filters[3], filters[2])
        self.Up_conv4 = pyrimad_decoder(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        # self.Up_conv3 = conv_block(filters[2], filters[1])
        self.Up_conv3 = pyrimad_decoder(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        # self.Up_conv2 = conv_block(filters[1], filters[0])
        self.Up_conv2 = pyrimad_decoder(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        # 光谱梯度
        self.spectralgradconv1 = spectralgradconv(in_ch, filters[0], 1, ks=5, stride=1, first_layer='False')
        self.spectralgradconv2 = spectralgradconv(filters[0], filters[1], 1, ks=5, stride=1, first_layer='False')
        self.spectralgradconv3 = spectralgradconv(filters[1], filters[2], 1, ks=5, stride=1, first_layer='False')
        self.spectralgradconv4 = spectralgradconv(filters[2], filters[3], 1, ks=5, stride=1, first_layer='False')
        self.spectralgradconv5 = spectralgradconv(filters[3], filters[4], 1, ks=5, stride=1, first_layer='False')

        self.cba5 = ChannelBoundaryAttention(filters[4])
        self.cba4 = ChannelBoundaryAttention(filters[3])
        self.cba3 = ChannelBoundaryAttention(filters[2])
        self.cba2 = ChannelBoundaryAttention(filters[1])




       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)
        se1 = self.spectralgradconv1(x)
        e1 = se1 + e1

        e2 = self.Maxpool1(e1)
        se2 = self.spectralgradconv2(e2)
        e2 = self.Conv2(e2)
        e2 = se2 + e2

        e3 = self.Maxpool2(e2)
        se3 = self.spectralgradconv3(e3)
        e3 = self.Conv3(e3)
        e3 = se3 + e3

        e4 = self.Maxpool3(e3)
        se4 = self.spectralgradconv4(e4)
        e4 = self.Conv4(e4)
        e4 = se4 + e4

        e5 = self.Maxpool4(e4)
        se5 = self.spectralgradconv5(e5)
        e5 = self.Conv5(e5)
        e5 = se5 + e5


        d5 = self.Up5(e5)
        d5 = self.cba5(e4, d5)
        # d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.cba4(e3, d4)
        # d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.cba3(e2, d3)
        # d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.cba2(e1, d2)
        # d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)

        return out



if __name__ == '__main__':

    from thop import profile
    from thop import clever_format
    rgb = torch.randn(4, 3, 256, 256)

    net = U_Net(in_ch=3, out_ch=2)
    # print(stat(net, (3, 512, 512)))
    out = net(rgb)
    flops, params = profile(net, inputs=(rgb,))
    flops, params = clever_format([flops, params], '%.3f')

    print(out.shape)
    print(f"运算量：{flops}, 参数量：{params}")


    print(out.shape)