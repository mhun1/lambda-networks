import torch
from torch import nn
from lambda_networks import LambdaLayer


class DoubleLambda(nn.Module):
    def __init__(self, dim_i, dim_o, r):
        super(DoubleLambda, self).__init__()

        self.first = LambdaLayer(dim=dim_i, dim_out=dim_o, r=r, dim_k=16, heads=4, dim_u=1)
        self.second = LambdaLayer(dim=dim_o, dim_out=dim_o, r=r, dim_k=16, heads=4, dim_u=1)

    def forward(self, x):
        return self.second(self.first(x))


class UpLambda(nn.Module):
    def __init__(self, dim_i, dim_o, r):
        super(UpLambda, self).__init__()
        self.shrink = LambdaLayer(dim=dim_i, dim_out=dim_o, r=r, dim_k=16, heads=4, dim_u=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, x2):
        x = self.shrink(x)
        x = self.up(x)
        x = torch.cat([x, x2], dim=1)
        return x


class Lambda_Unet(nn.Module):
    def __init__(self,n_classes):
        super(Lambda_Unet, self).__init__()

        self.n_classes = n_classes
        self.max_pool = nn.MaxPool2d((2, 2))
        self.up_lambda = UpLambda(1024, 512, 7)
        self.up_lambda2 = UpLambda(512, 256, 7)
        self.up_lambda3 = UpLambda(256, 128, 7)
        self.up_lambda4 = UpLambda(128, 64, 7)

        self.lambda_conv = DoubleLambda(1, 64, 7)
        self.lambda_conv2 = DoubleLambda(64, 128, 7)
        self.lambda_conv3 = DoubleLambda(128, 256, 7)
        self.lambda_conv4 = DoubleLambda(256, 512, 7)
        self.lambda_conv5 = DoubleLambda(512, 1024, 7)

        self.lambda_reverse = DoubleLambda(1024, 512, 7)
        self.lambda_reverse2 = DoubleLambda(512, 256, 7)
        self.lambda_reverse3 = DoubleLambda(256, 128, 7)

        self.same_lambda = DoubleLambda(128, 64, 7)
        self.output_lambda = LambdaLayer(dim=64, dim_out=self.n_classes, r=7, dim_k=16, heads=1, dim_u=1)

    def forward(self, x):
        print("INPUT SHAPE: {}".format(x.shape))

        x_1 = self.lambda_conv(x)  # (1,64,512,512)
        x = self.max_pool(x_1)  # (1,64,256,256)

        x_2 = self.lambda_conv2(x)  # (1,128,256,256)
        x = self.max_pool(x_2)  # (1,128,128,128)

        x_3 = self.lambda_conv3(x)  # (1,256,128,128)
        x = self.max_pool(x_3)  # (1,256,64,64)

        x_4 = self.lambda_conv4(x)  # (1,512,64,64)
        x = self.max_pool(x_4)  # (1,512,32,32)

        x = self.lambda_conv5(x)  # (1,1024,32,32)

        x = self.up_lambda(x, x_4)  # (1,1024,64,64)

        x = self.lambda_reverse(x)  # (1,512,64,64)

        x = self.up_lambda2(x, x_3)  # (1,512,128,128)

        x = self.lambda_reverse2(x)  # (1,256,128,128)

        x = self.up_lambda3(x, x_2)  # (1,256,256,256)

        x = self.lambda_reverse3(x)  # (1,128,256,256)

        x = self.up_lambda4(x, x_1)  # (1,128,512,512)

        x = self.same_lambda(x)  # (1,64,512,512)

        x = self.output_lambda(x)  # (1,2,512,512)

        return x




