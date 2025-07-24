import openvino
"""import torch
from torch import nn
from torch.nn import init





class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        return x * y.expand_as(x)




class BasicConv(nn.Module):
    def __init__(
            self,
            in_planes,
            out_planes,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            relu=True,
            bn=True,
            bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes

        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )

        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )



class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale



class TripletAttention(nn.Module):
    def __init__(
            self,
            gate_channels,
            reduction_ratio=16,
            pool_types=["avg", "max"],
            no_spatial=False,
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out




class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)



class TransConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(TransConv, self).__init__()
        self.tranconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch,out_ch,kernel_size=2,stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self,input):
        return self.tranconv(input)


class DRA_net(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DRA_net, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.w1 = nn.Conv2d(in_ch, 64, kernel_size=1, padding=0, stride=1) # to increase the dimensions
        self.ecaa1 = ECAAttention(kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = DoubleConv(64, 128)
        self.w2 = nn.Conv2d(64, 128, kernel_size=1, padding=0, stride=1) # to increase the dimensions
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = DoubleConv(128, 256)
        self.w3 = nn.Conv2d(128, 256, kernel_size=1, padding=0, stride=1) # to increase the dimensions
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = DoubleConv(256, 512)
        self.w4 = nn.Conv2d(256, 512, kernel_size=1, padding=0, stride=1) # to increase the dimensions
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.conv5 = DoubleConv(512, 1024)
        self.w5 = nn.Conv2d(512, 1024, kernel_size=1, padding=0, stride=1) # to increase the dimensions
        #self.up6 = nn.ConvTranspose2d(1024, 512, image, stride=image)
        self.up6 = TransConv(1024, 512)
        self.tr6 =TripletAttention(gate_channels=512)
        self.conv6 = DoubleConv(1024, 512)
        #self.up7 = nn.ConvTranspose2d(512, 256, image, stride=image)
        self.up7 = TransConv(512, 256)
        self.tr7 = TripletAttention(gate_channels=256)
        self.conv7 = DoubleConv(512, 256)
        # self.up8 = nn.ConvTranspose2d(256, 128, image, stride=image)
        self.up8 = TransConv(256, 128)
        self.tr8 = TripletAttention(gate_channels=128)
        self.conv8 = DoubleConv(256, 128)
        #self.up9 = nn.ConvTranspose2d(128, 64, image, stride=image)
        self.up9 = TransConv(128, 64)
        self.tr9 = TripletAttention(gate_channels=64)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
   
        self.dropout = nn.Dropout2d(p=0.5)
        self.softmax = nn.Softmax()



    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        r1 = self.w1(x)  # residual block，
        c1 = self.conv1(x)+r1
        e1 = self.ecaa1(c1)
        #print("e1:", e1.shape)
        p1 = self.pool1(e1)
        #print("p1:", p1.shape)
        r2 = self.w2(p1)
        c2 = self.conv2(p1)+r2
        e2 = self.ecaa1(c2)
        #print("r2:", r2.shape)
        p2 = self.pool2(e2)
        #v2 = self.vit2 (e2)
        # print("p2:", p2.shape)
        r3 = self.w3(p2)
        c3 = self.conv3(p2)+r3
        e3 = self.ecaa1(c3)
        p3 = self.pool3(e3)
        # print("p3:", p3.shape)
        r4 = self.w4(p3)
        c4 = self.conv4(p3)+r4
        e4 = self.ecaa1(c4)
        mid1 = self.dropout(e4)
        p4 = self.pool4(mid1)
        # print("p4:", p4.shape)
        r5 = self.w5(p4)
        c5 = self.conv5(p4)+r5
        e5 = self.ecaa1(c5)
        mid2 = self.dropout(e5)
        up_6 = self.up6(mid2)
        # print("up_6:", up_6.shape)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        e6 = self.tr6(c6)
        up_7 = self.up7(e6)
        #print("up_7:", up_7.shape)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        e7 = self.tr7(c7)
        up_8 = self.up8(e7)
        # print("up_8:", up_8.shape)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        e8 = self.tr8(c8)
        up_9 = self.up9(e8)
        # print("up_9:", up_9.shape)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        #e9 = self.tr9(c9)
        #print("c9",c9.shape)
        c10 = self.conv10(c9)
        e10 = self.ecaa1(c10)
        #print("e10",e10.shape)
        return e10



#==================================================================================
if __name__ == '__main__':
    input = torch.rand(16, 3, 256, 256)
    unet = DRA_net(3, 8)
    out = unet(input)
    print(out.size())
"""
import torch
import torch.nn as nn
import torch.nn.init as init

class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        return x * y.expand_as(x)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                             stride=stride, padding=padding, dilation=dilation,
                             groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), 
             torch.mean(x, 1).unsqueeze(1)),
            dim=1
        )

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, padding=(kernel_size-1)//2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"], no_spatial=False):
        super().__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            return (x_out + x_out11 + x_out21) / 3
        return (x_out11 + x_out21) / 2

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class TransConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.tranconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.tranconv(x)

class DRA_net(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        # Encoder
        self.encoder = nn.ModuleDict({
            'conv1': DoubleConv(in_ch, 64),
            'conv2': DoubleConv(64, 128),
            'conv3': DoubleConv(128, 256),
            'conv4': DoubleConv(256, 512),
            'conv5': DoubleConv(512, 1024)
        })
        
        self.w = nn.ModuleList([
            nn.Conv2d(in_ch, 64, 1),
            nn.Conv2d(64, 128, 1),
            nn.Conv2d(128, 256, 1),
            nn.Conv2d(256, 512, 1),
            nn.Conv2d(512, 1024, 1)
        ])
        
        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(4)])
        self.ecaa = nn.ModuleList([ECAAttention(3) for _ in range(5)])
        self.dropout = nn.Dropout2d(0.5)
        
        # Decoder
        self.up = nn.ModuleList([
            TransConv(1024, 512),
            TransConv(512, 256),
            TransConv(256, 128),
            TransConv(128, 64)
        ])
        
        self.tr = nn.ModuleList([
            TripletAttention(512),
            TripletAttention(256),
            TripletAttention(128),
            TripletAttention(64)
        ])
        
        self.decoder = nn.ModuleDict({
            'conv6': DoubleConv(1024, 512),
            'conv7': DoubleConv(512, 256),
            'conv8': DoubleConv(256, 128),
            'conv9': DoubleConv(128, 64)
        })
        
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        self.softmax = nn.Softmax(dim=1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        skip_connections = []
        
        for i in range(5):
            r = self.w[i](x if i == 0 else p)
            c = self.encoder[f'conv{i+1}'](x if i == 0 else p) + r
            e = self.ecaa[i](c)
            
            if i < 4:
                p = self.pools[i](e)
                skip_connections.append(e)
            else:
                mid = self.dropout(e)
        
        # Decoder
        for i in range(4):
            up = self.up[i](mid if i == 0 else e_dec)
            merge = torch.cat([up, skip_connections[3-i]], dim=1)
            c_dec = self.decoder[f'conv{6+i}'](merge)
            e_dec = self.tr[i](c_dec)
        
        # Output
        out = self.conv10(e_dec)
        return self.softmax(out)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DRA_net(3, 8).to(device)
    input = torch.rand(16, 3, 256, 256).to(device)
    out = model(input)
    print(out.size())  # Should be torch.Size([16, 8, 256, 256])