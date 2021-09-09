import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels,out_channels,kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv

def crop_tensor(tensor, target_tensor):
    # tensor is [bs,c,h,w] with h==w
    assert(target_tensor.size()[2]==target_tensor.size()[3])
    assert(tensor.size()[2]==tensor.size()[3])
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = (tensor_size - target_size) // 2
    return tensor[:,:,delta:tensor_size-delta,delta:tensor_size-delta]

class UNet(nn.Module):
    def __init__(self,n_channels,n_classes,bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv_1 = double_conv(n_channels,64)
        self.down_conv_2 = double_conv(64,128)
        self.down_conv_3 = double_conv(128,256)
        self.down_conv_4 = double_conv(256,512)
        self.down_conv_5 = double_conv(512,1024)

        self.up_trans_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv_1 = double_conv(1024, 512)
        self.up_trans_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv_2 = double_conv(512, 256)
        self.up_trans_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv_3 = double_conv(256, 128)
        self.up_trans_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv_4 = double_conv(128, 64)
        self.output = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, image):
        # image is [bs,c,h,w]

        # encoder
        x1 = self.down_conv_1(image) #
        x = self.max_pool_2x2(x1)
        x2 = self.down_conv_2(x) #
        x = self.max_pool_2x2(x2)
        x3 = self.down_conv_3(x) #
        x = self.max_pool_2x2(x3)
        x4 = self.down_conv_4(x) #
        x = self.max_pool_2x2(x4)
        x = self.down_conv_5(x)

        # decoder
        x = self.up_trans_1(x)
        y = crop_tensor(x4,x)
        x = self.up_conv_1(torch.cat([y,x],1))
        
        x = self.up_trans_2(x)
        y = crop_tensor(x3,x)
        print(x.size())
        print(x3.size())
        print(y.size())
        x = self.up_conv_2(torch.cat([y,x],1))

        x = self.up_trans_3(x)
        y = crop_tensor(x2,x)
        x = self.up_conv_3(torch.cat([y,x],1))

        x = self.up_trans_4(x)
        y = crop_tensor(x1,x)
        x = self.up_conv_4(torch.cat([y,x],1))

        x = self.output(x)
        print(x.size())
        return x

if __name__=="__main__":
    net = UNet(3,1)
    image = torch.rand((1,3,572,572))
    print(net(image))