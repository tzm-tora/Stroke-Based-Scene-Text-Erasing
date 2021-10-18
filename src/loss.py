import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
import cfg


class InpaintingLoss(nn.Module):
    def __init__(self):
        super(InpaintingLoss, self).__init__()
        self.l1 = nn.L1Loss()
        # default extractor is VGG16
        self.extractor = VGG16FeatureExtractor()

    def forward(self, input, o_mask, t_mask, output, gt):
        # Dice Loss
        dice_loss = build_dice_loss(t_mask, o_mask)
        mask_l1_loss = self.l1(o_mask, t_mask)

        # Non-hole pixels directly set to ground truth
        comp = o_mask * input + (1 - o_mask) * output

        # Total Variation Regularization
        tv_loss = total_variation_loss(comp, o_mask)

        # Hole Pixel Loss
        hole_loss = self.l1((1-o_mask) * output, (1-o_mask) * gt)

        # Valid Pixel Loss
        valid_loss = self.l1(o_mask * output, o_mask * gt)

        comp_l1_loss = self.l1(comp, gt)

        # Perceptual Loss and Style Loss
        feats_out = self.extractor(output)
        feats_comp = self.extractor(comp)
        feats_gt = self.extractor(gt)
        perc_loss = 0.0
        style_loss = 0.0
        # Calculate the L1Loss for each feature map
        for i in range(3):
            perc_loss += self.l1(feats_out[i], feats_gt[i])
            perc_loss += self.l1(feats_comp[i], feats_gt[i])
            style_loss += self.l1(gram_matrix(feats_out[i]),
                                  gram_matrix(feats_gt[i]))
            style_loss += self.l1(gram_matrix(feats_comp[i]),
                                  gram_matrix(feats_gt[i]))

        mask_l1_loss *= cfg.dice_coef
        dice_loss *= cfg.dice_coef
        valid_loss *= cfg.valid_coef
        hole_loss *= cfg.hole_coef
        perc_loss *= cfg.perc_coef
        style_loss *= cfg.style_coef
        tv_loss *= cfg.tv_coef
        loss = dice_loss + mask_l1_loss + valid_loss + \
            hole_loss + perc_loss + style_loss + comp_l1_loss + tv_loss

        return loss, {'mask_l1': mask_l1_loss, 'dice': dice_loss, 'valid': valid_loss, 'hole': hole_loss, 'comp_l1': comp_l1_loss, 'perc': perc_loss, 'style': style_loss, 'tv': tv_loss}


# The network of extracting the feature for perceptual and style loss
class VGG16FeatureExtractor(nn.Module):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    vgg_mean = [0.485, 0.456, 0.406]
    vgg_std = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        normalization = Normalization(
            self.mean, self.std, self.vgg_mean, self.vgg_std)
        # Define the each feature exractor
        self.enc_1 = nn.Sequential(normalization, *vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{}'.format(i+1)).parameters():
                param.requires_grad = False

    def forward(self, input):
        feature_maps = [input]
        for i in range(3):
            feature_map = getattr(self, 'enc_{}'.format(i+1))(feature_maps[-1])
            feature_maps.append(feature_map)
        return feature_maps[1:]


# Normalization Layer for VGG
class Normalization(nn.Module):
    def __init__(self, mean, std, vgg_mean, vgg_std):
        super(Normalization, self).__init__()
        # .view the vgg_mean and vgg_std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.vgg_mean = torch.tensor(vgg_mean).view(-1, 1, 1)
        self.vgg_std = torch.tensor(vgg_std).view(-1, 1, 1)

    def forward(self, input):
        # normalize img
        if self.vgg_mean.type() != input.type():
            self.mean = self.mean.to(input)
            self.std = self.std.to(input)
            self.vgg_mean = self.vgg_mean.to(input)
            self.vgg_std = self.vgg_std.to(input)
        return ((input * self.std + self.mean) - self.vgg_mean) / self.vgg_std


# Calcurate the Gram Matrix of feature maps
def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image, o_mask, method='mean'):
    hole_mask = 1 - o_mask
    colomns_in_Pset = hole_mask[:, :, :, 1:] * hole_mask[:, :, :, :-1]
    rows_in_Pset = hole_mask[:, :, 1:, :] * hole_mask[:, :, :-1:, :]
    if method == 'sum':
        loss = torch.sum(torch.abs(colomns_in_Pset*(
            image[:, :, :, 1:] - image[:, :, :, :-1]))) + \
            torch.sum(torch.abs(rows_in_Pset*(
                image[:, :, :1, :] - image[:, :, -1:, :])))
    else:
        loss = torch.mean(torch.abs(colomns_in_Pset*(
            image[:, :, :, 1:] - image[:, :, :, :-1]))) + \
            torch.mean(torch.abs(rows_in_Pset*(
                image[:, :, :1, :] - image[:, :, -1:, :])))
    return loss


def build_dice_loss(x_t, x_o):
    epsilon = 1e-8
    N = x_t.shape[0]
    x_t_flat = x_t.view(N, -1)
    x_o_flat = x_o.view(N, -1)
    intersection = (x_t_flat * x_o_flat).sum(1)  # N, H*W -> N, 1 -> scolar
    union = x_t_flat.sum(1) + x_o_flat.sum(1)
    dice_loss = 1. - ((2. * intersection + epsilon)/(union + epsilon)).mean()
    return dice_loss
