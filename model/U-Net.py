# U-Net Architecture
# Contracting path - Conv(3x3 kernel, stride:1) x 2
# Bottle neck - Conv(3x3 kernel, stride: 1) x 2
# Expanding path - Up-Conv(2x2kernel, stride:2) after copy and crop Conv(3x3 kernel, stride:1)x2


####### import #######
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
####### import ######




########  UNet  #########
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # (Convoultion + BatchNormalization + ReLU)
        # BatchNormalization 사용 이유: 신경망의 각 레이어는 학습 중에 입력 분포가 변화하기 때문에 이러한 변화는
        # 학습을 불안정하게 하고 새로운 입력 분포에 적응하기 어렵다. 각 레이어의 입력을 정규화하므로 변화를 줄임.
        def CBR2d(input_channel, output_channel, kernel_size=3, stride=1):
            layer = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size= kernel_size, stride=stride),
                nn.BatchNorm2d(num_features=output_channel),
                nn.ReLU()
            )
            return layer

        # Contracting path
        # layer1 1 -> 64
        self.conv1 = nn.Sequential(CBR2d(1, 64), CBR2d(64, 64))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer2 64 ->128
        self.conv2 = nn.Sequential(CBR2d(64, 128), CBR2d(128, 128))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer3 128 -> 256
        self.conv3 = nn.Sequential(CBR2d(128, 256), CBR2d(256, 256))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer4 256 -> 512
        self.conv4 = nn.Sequential(CBR2d(256, 512), CBR2d(512, 512), nn.Dropout(p=0.5)) # Dropout 과적합 방지
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        # Bottleneck
        # 512 -> 1024
        self.bottleNeck = nn.Sequential(CBR2d(512, 1024), CBR2d(1024, 1024))


        # Expanding path
        # 1nd UpSampling 1024 -> 512
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.ex_conv1 = nn.Sequential(CBR2d(1024, 512), CBR2d(512, 512))

        # 2nd UpSampling 512 -> 256
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.ex_conv2 = nn.Sequential(CBR2d(512, 256), CBR2d(256, 256))

        # 3nd UpSampling 256 -> 128
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.ex_conv3 = nn.Sequential(CBR2d(256, 128), CBR2d(128, 128))

        # 4nd UpSampling 128 -> 64
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.ex_conv4 = nn.Sequential(CBR2d(128, 64), CBR2d(64, 64))


        # Final Layer
        self.fc = nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, x):
        # Contracting path
        layer1 = self.conv1(x)
        out = self.pool1(layer1)

        layer2 = self.conv2(out)
        out = self.pool2(layer2)

        layer3 = self.conv3(out)
        out = self.pool3(layer3)

        layer4 = self.conv4(out)
        out = self.pool4(layer4)

        # bottleNeck
        bottleneck = self.bottleNeck(out)


        # Expanding path
        # Center_crop: layer를 upconv의 출력크기만큼 중앙 크롭하여 결합
        upconv1 = self.upconv1(bottleneck)
        cat1 = torch.cat((F.center_crop(layer4, [upconv1.shape[2], upconv1.shape[3]]), upconv1), dim=1)
        ex_layer1 = self.ex_conv1(cat1)

        upconv2 = self.upconv2(ex_layer1)
        cat2 = torch.cat((F.center_crop(layer3, [upconv2.shape[2], upconv2.shape[3]]), upconv2), dim=1)
        ex_layer2 = self.ex_conv2(cat2)

        upconv3 = self.upconv3(ex_layer2)
        cat3 = torch.cat((F.center_crop(layer2, [upconv3.shape[2], upconv3.shape[3]]), upconv3), dim=1)
        ex_layer3 = self.ex_conv3(cat3)

        upconv4 = self.upconv4(ex_layer3)
        cat4 = torch.cat((F.center_crop(layer1, [upconv4.shape[2], upconv4.shape[3]]), upconv4), dim=1)
        ex_layer4 = self.ex_conv4(cat4)

        # output layer
        out = self.fc(ex_layer4)
        return out


# model = UNet()
# input_image = torch.randn(1, 1, 572, 572)  # (batch_size, channels, height, width)
# output = model(input_image)
# print(f"출력 크기: {output.shape}")  # torch.Size([1, 1, 388, 388])