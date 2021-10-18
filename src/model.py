import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        nn.init.constant_(m[-1].weight, val=0)
        m[-1].inited = True
    else:
        nn.init.constant_(m.weight, val=0)
        m.inited = True


class ContextBlock2d(nn.Module):

    def __init__(self, inplanes, planes, pool='att', fusions=['channel_add'], ratio=4):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(
                inplanes, 1, kernel_size=1)  # context Modeling
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #---------------------------------------------------------------------#
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        #---------------------------------------------------------------------#
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            nn.init.kaiming_uniform_(self.conv_mask.weight, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)
    #---------------------------------------------------------------------#

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W] 添加一个维度
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)  # softmax操作
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
    #---------------------------------------------------------------------#
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class build_attention_block(nn.Module):

    def __init__(self):
        super(build_attention_block, self).__init__()
        self.layer = self._make_layer1()

    def _make_layer1(self):
        layers = []
        layers.append(ContextBlock2d(512, 512))
        # layers.append(NonLocalBlock2D(in_chs=512))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        return x


class PartialConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        # Inherit the parent class (Conv2d)
        super(PartialConv2d, self).__init__(in_channels, out_channels,
                                            kernel_size, stride=stride,
                                            padding=padding, dilation=dilation,
                                            groups=groups, bias=bias,
                                            padding_mode=padding_mode)
        # Define the kernel for updating mask
        self.mask_kernel = torch.ones(self.out_channels, self.in_channels,
                                      self.kernel_size[0], self.kernel_size[1])
        # Define sum1 for renormalization
        self.sum1 = self.mask_kernel.shape[1] * self.mask_kernel.shape[2] \
                                              * self.mask_kernel.shape[3]
        # Define the updated mask
        self.update_mask = None
        # Define the mask ratio (sum(1) / sum(M))
        self.mask_ratio = None
        # Initialize the weights for image convolution
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, img, mask):
        with torch.no_grad():
            if self.mask_kernel.type() != img.type():
                self.mask_kernel = self.mask_kernel.to(img)
            # Create the updated mask
            # for calcurating mask ratio (sum(1) / sum(M))
            self.update_mask = F.conv2d(mask, self.mask_kernel,
                                        bias=None, stride=self.stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        groups=1)
            # calcurate mask ratio (sum(1) / sum(M))
            self.mask_ratio = self.sum1 / (self.update_mask + 1e-8)
            self.update_mask = torch.clamp(self.update_mask, 0, 1)
            self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # calcurate WT . (X * M)
        conved = torch.mul(img, mask)
        conved = F.conv2d(conved, self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)

        if self.bias is not None:
            # Maltuply WT . (X * M) and sum(1) / sum(M) and Add the bias
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(conved - bias_view, self.mask_ratio) + bias_view
            # The masked part pixel is updated to 0
            output = torch.mul(output, self.mask_ratio)
        else:
            # Multiply WT . (X * M) and sum(1) / sum(M)
            output = torch.mul(conved, self.mask_ratio)

        return output, self.update_mask


class UpsampleConcat(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the upsampling layer with nearest neighbor
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, dec_feature, enc_feature, dec_mask, enc_mask):
        # upsample and concat features
        out = self.upsample(dec_feature)
        out = torch.cat([out, enc_feature], dim=1)
        # upsample and concat masks
        out_mask = self.upsample(dec_mask)
        out_mask = torch.cat([out_mask, enc_mask], dim=1)
        return out, out_mask


class PConvActiv(nn.Module):
    def __init__(self, in_ch, out_ch, sample='none-3', dec=False,
                 bn=True, active='relu', conv_bias=False):
        super().__init__()
        # Define the partial conv layer
        if sample == 'down-7':
            params = {"kernel_size": 7, "stride": 2, "padding": 3}
        elif sample == 'down-7-1':
            params = {"kernel_size": 7, "stride": 1, "padding": 3}
        elif sample == 'down-5':
            params = {"kernel_size": 5, "stride": 2, "padding": 2}
        elif sample == 'down-5-1':
            params = {"kernel_size": 5, "stride": 1, "padding": 2}
        elif sample == 'down-3':
            params = {"kernel_size": 3, "stride": 2, "padding": 1}
        else:
            params = {"kernel_size": 3, "stride": 1, "padding": 1}
        self.partialconv = PartialConv2d(
            in_ch, out_ch, params["kernel_size"], params["stride"], params["padding"], bias=conv_bias)

        # Define other layers
        if dec:
            self.upcat = UpsampleConcat()
        if bn:
            bn = nn.BatchNorm2d(out_ch)
        if active == 'relu':
            self.activation = nn.ReLU()
        elif active == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif active == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, img, mask, enc_img=None, enc_mask=None):
        if hasattr(self, 'upcat'):
            out, update_mask = self.upcat(img, enc_img, mask, enc_mask)
            out, update_mask = self.partialconv(out, update_mask)
        else:
            out, update_mask = self.partialconv(img, mask)
        if hasattr(self, 'bn'):
            out = self.bn(out)
        if hasattr(self, 'activation'):
            out = self.activation(out)
        return out, update_mask


class PConvUNet(nn.Module):
    def __init__(self, finetune, in_ch=3, layer_size=6):
        super().__init__()
        self.freeze_enc_bn = True if finetune else False
        self.layer_size = layer_size

        # 3, 128, 128*5 -> 64, 64, 64
        self.enc_1 = PConvActiv(in_ch, 64, 'down-7', bn=False)
        # 64, 64, 64*5 -> 64, 64, 64
        self.enc_2 = PConvActiv(64,  64, 'down-3-1')
        # 64, 64, 64*5 -> 64, 64, 64
        self.enc_3 = PConvActiv(64,  64, 'down-3-1')
        # 64, 64, 64*5 -> 128, 32, 32
        self.enc_4 = PConvActiv(64, 128, 'down-5')
        self.enc_5 = PConvActiv(128, 128, 'down-3-1')
        self.enc_6 = PConvActiv(128, 128, 'down-3-1')
        # 128, 32, 32 -> 256, 16, 16
        self.enc_7 = PConvActiv(128, 256, 'down-3')
        self.enc_8 = PConvActiv(256, 256, 'down-3-1')
        self.enc_9 = PConvActiv(256, 256, 'down-3-1')  # 256, 16, 16

        self.fuse_0 = _conv_bn_relu(512, 256)
        self.attention_block = build_attention_block()
        self.dec_9 = PConvActiv(256+128, 256, dec=True, active='leaky')
        self.dec_8 = PConvActiv(256, 256, dec=False, active='leaky')
        self.dec_7 = PConvActiv(256, 128, dec=False, active='leaky')
        self.dec_6 = PConvActiv(128+64, 128, dec=True, active='leaky')
        self.dec_5 = PConvActiv(128, 128, dec=False, active='leaky')
        self.dec_4 = PConvActiv(128, 64,  dec=False, active='leaky')
        self.dec_3 = PConvActiv(64 + 3, 64,  dec=True, active='leaky')
        self.dec_2 = PConvActiv(64, 64,  dec=False, active='leaky')
        self.dec_1 = PConvActiv(
            64, 3,   dec=False, bn=False, active='tanh', conv_bias=True)

    def forward(self, img, mask, fea_m):

        d1, update_mask1 = self.enc_1(img, mask)
        d2, update_mask2 = self.enc_2(d1, update_mask1)
        d3, update_mask3 = self.enc_3(d2, update_mask2)
        d4, update_mask4 = self.enc_4(d3, update_mask3)
        d5, update_mask5 = self.enc_5(d4, update_mask4)
        d6, update_mask6 = self.enc_6(d5, update_mask5)
        d7, update_mask7 = self.enc_7(d6, update_mask6)
        d8, update_mask8 = self.enc_8(d7, update_mask7)
        d9, update_mask9 = self.enc_9(d8, update_mask8)
        u0 = torch.cat([d9, fea_m], dim=1)
        u0 = self.attention_block(u0)
        u0 = self.fuse_0(u0)
        # print(u0.shape, d9.shape)
        u1, update_masku1 = self.dec_9(
            u0, update_mask9, d6, update_mask6)  # dec=True
        u2, update_masku2 = self.dec_8(u1, update_masku1)
        u3, update_masku3 = self.dec_7(u2, update_masku2)
        u4, update_masku4 = self.dec_6(
            u3, update_masku3, d3, update_mask3)  # dec=True
        u5, update_masku5 = self.dec_5(u4, update_masku4)
        u6, update_masku6 = self.dec_4(u5, update_masku5)
        u7, update_masku7 = self.dec_3(
            u6, update_masku6, img, mask)  # dec=True
        u8, update_masku8 = self.dec_2(u7, update_masku7)
        u9, update_masku9 = self.dec_1(u8, update_masku8)

        return u9, update_masku9

    def train(self, mode=True):
        """Override the default train() to freeze the BN parameters
        In initial training, BN set to True
        In fine-tuning stage, BN set to False
        """
        super().train(mode)
        if not self.freeze_enc_bn:
            return
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                module.eval()


class _res_block(nn.Module):

    def __init__(self, in_channels=256, out_channels=256):
        super(_res_block, self).__init__()
        self.med_channel = 64
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=self.med_channel, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=self.med_channel,
                               out_channels=self.med_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.med_channel,
                               out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        identity = x  # 分支输出
        out = self.conv1(x)
        out = self.leakyrelu(out)
        out = self.conv2(out)
        out = self.leakyrelu(out)
        out = self.conv3(out)
        out += identity
        out = self.bn(out)
        out = self.leakyrelu(out)
        return out


class _conv_bn_relu(nn.Module):  # 只改变channels

    def __init__(self, in_channels, out_channels):
        super(_conv_bn_relu, self).__init__()
        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                            bias=False)]
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class build_res_net(nn.Module):

    def __init__(self):
        super(build_res_net, self).__init__()
        self.layer = self._make_layer(4)

    def _make_layer(self, block_num):
        layers = []
        for _ in range(1, block_num):
            layers.append(_res_block(in_channels=256,
                                     out_channels=256))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        return x


class build_encoder_net(nn.Module):

    def __init__(self, in_channels=3, out_channels=256, get_feature_map=False):
        super(build_encoder_net, self).__init__()
        self.get_encoder_feature_map = get_feature_map
        self.down1 = _conv_bn_relu(
            in_channels=in_channels, out_channels=32) 
        self.down2 = _conv_bn_relu(in_channels=32, out_channels=32)  
        self.down3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.down4 = _conv_bn_relu(in_channels=64, out_channels=64)  
        self.down5 = _conv_bn_relu(in_channels=64, out_channels=64)  
        self.down6 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.down7 = _conv_bn_relu(in_channels=128, out_channels=128)  
        self.down8 = _conv_bn_relu(in_channels=128, out_channels=128)  
        self.down9 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.down10 = _conv_bn_relu(in_channels=256, out_channels=256) 
        self.down11 = _conv_bn_relu(in_channels=256, out_channels=256)  
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.down1(x)
        out = self.down2(out)
        f0 = out
        out = self.down3(out)
        out = self.leakyrelu(out)
        out = self.down4(out)
        out = self.down5(out)
        f1 = out  # 64,
        out = self.down6(out)
        out = self.leakyrelu(out)
        out = self.down7(out)
        out = self.down8(out)
        f2 = out  # 128,
        out = self.down9(out)
        out = self.leakyrelu(out)
        out = self.down10(out)
        out = self.down11(out)
        if self.get_encoder_feature_map:
            return out, [f2, f1, f0]
        else:
            return out


class build_decoder_net(nn.Module):
    def __init__(self, in_channels_1=256, in_channels_2=256, in_channels_3=128, in_channels_4=64, out_channels=32):
        super(build_decoder_net, self).__init__()
        self.up1 = _conv_bn_relu(
            in_channels=in_channels_1, out_channels=256)  # ??
        self.up2 = _conv_bn_relu(in_channels=256, out_channels=256)  # ?????
        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.up4 = _conv_bn_relu(
            in_channels=in_channels_2, out_channels=128)  # ?????
        self.up5 = _conv_bn_relu(in_channels=128, out_channels=128)  # ?????
        self.up6 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.up7 = _conv_bn_relu(
            in_channels=in_channels_3, out_channels=64)  # ?????
        self.up8 = _conv_bn_relu(in_channels=64, out_channels=64)  # ?????
        self.up9 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1,
                                      output_padding=1)
        self.up10 = _conv_bn_relu(
            in_channels=in_channels_4, out_channels=32)  # ?????
        self.up11 = _conv_bn_relu(in_channels=32, out_channels=32)  # ?????
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, fuse):
        # if fuse and fuse[0] is not None:
        #     x = torch.cat((x, fuse[0]), 1)   # 512, 28, 28
        out = self.up1(x)  # bnrelu
        out = self.up2(out)  # bnrelu
        out = self.up3(out)
        out = self.leakyrelu(out)
        if fuse and fuse[0] is not None:
            out = torch.cat((out, fuse[0]), 1)
        out = self.up4(out)  # bnrelu
        out = self.up5(out)  # bnrelu
        out = self.up6(out)
        out = self.leakyrelu(out)
        if fuse and fuse[1] is not None:
            out = torch.cat((out, fuse[1]), 1)
        out = self.up7(out)  # bnrelu
        out = self.up8(out)  # bnrelu
        out = self.up9(out)
        out = self.leakyrelu(out)
        if fuse and fuse[2] is not None:
            out = torch.cat((out, fuse[2]), 1)
        out = self.up10(out)  # bnrelu
        out = self.up11(out)  # bnrelu
        return out


class build_mask_prediction_net(nn.Module):

    def __init__(self):
        super(build_mask_prediction_net, self).__init__()
        self.encoder = build_encoder_net(
            in_channels=3, out_channels=256, get_feature_map=True)
        self.resblock = build_res_net()
        self.decoder = build_decoder_net(
            in_channels_1=256, in_channels_2=256, in_channels_3=128, in_channels_4=64, out_channels=32)
        layers_sig = []
        layers_sig.append(nn.Conv2d(in_channels=32, out_channels=3,
                                    kernel_size=3, stride=1, padding=1, bias=True))
        layers_sig.append(nn.Sigmoid())
        self.sigmiod_out = nn.Sequential(*layers_sig)

    def forward(self, x_s):
        x_s, skip_cons = self.encoder(x_s)  # 256, 28, 28
        x_s = self.resblock(x_s)  # 256, 28, 28
        y_s = self.decoder(x_s, skip_cons)  # 32, 224, 224
        o_mask = self.sigmiod_out(y_s)
        return o_mask, x_s


class build_generator(nn.Module):

    def __init__(self, finetune=False):
        super(build_generator, self).__init__()
        self.build_mask_prediction_net = build_mask_prediction_net()
        self.background_inpainting_net = PConvUNet(finetune=finetune)

    def forward(self, i_s):
        o_mask, fea_m = self.build_mask_prediction_net(i_s)
        o_b, _ = self.background_inpainting_net(i_s, o_mask, fea_m)
        return o_b, o_mask
