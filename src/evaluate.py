from skimage import metrics
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import cfg
import os
import numpy as np
import torch
import cv2
from PIL import Image


class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Line(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2


def GetLinePara(line):
    line.a = line.p1.y - line.p2.y
    line.b = line.p2.x - line.p1.x
    line.c = line.p1.x * line.p2.y - line.p2.x * line.p1.y


def GetCrossPoint(l1, l2):
    GetLinePara(l1)
    GetLinePara(l2)
    d = l1.a * l2.b - l2.a * l1.b
    p = Point()
    p.x = (l1.b * l2.c - l2.b * l1.c)*1.0 / d
    p.y = (l1.c * l2.a - l2.c * l1.a)*1.0 / d
    return p


def cross_point(quad):
    x1, y1 = quad[0]
    x2, y2 = quad[1]
    x3, y3 = quad[2]
    x4, y4 = quad[3]
    # x1, y1, x2, y2, x3, y3, x4, y4 = quad
    p1 = Point(x1, y1)
    p3 = Point(x3, y3)
    line1 = Line(p1, p3)

    p2 = Point(x2, y2,)
    p4 = Point(x4, y4)
    line2 = Line(p2, p4)
    Pc = GetCrossPoint(line1, line2)
    return (Pc.x, Pc.y)


def tensor2jpg(tensor, W, H):
    tensor = tensor.detach().squeeze(0).cpu()
    tensor = (tensor * 0.5 + 0.5)
    jpg_img = transforms.ToPILImage()(tensor)
    jpg_img = jpg_img.resize((W, H), Image.BICUBIC)
    jpg_img = cv2.cvtColor(np.asarray(jpg_img), cv2.COLOR_RGB2BGR)
    return jpg_img


def tensor2jpg_mask(tensor, W, H):
    tensor = tensor.detach().squeeze(0).cpu()
    jpg_img = transforms.ToPILImage()(tensor)
    jpg_img = jpg_img.resize((W, H), Image.BICUBIC)
    jpg_img = cv2.cvtColor(np.asarray(jpg_img), cv2.COLOR_RGB2BGR)
    return jpg_img


def cal_mse(src, tar):
    src = src/255
    tar = tar/255
    mse = metrics.mean_squared_error(src, tar)
    return mse


def cal_psnr(src, tar):
    psnr = metrics.peak_signal_noise_ratio(src, tar, data_range=255)
    return psnr


def cal_ssim(src, tar):
    ssim = metrics.structural_similarity(
        src, tar, data_range=255, multichannel=True)
    return ssim*100


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def is_txt_file(filename):
    return any(filename.endswith(extension) for extension in [".txt"])


def expand_roi(Poly, img_width=512, img_height=512, ratio=0.2):
    c_pts = cross_point(Poly)
    (tl, tr, br, bl) = Poly
    expand_poly = np.zeros((4, 2), dtype="float32")
    for i, coor in enumerate(Poly):
        cha_x, cha_y = (coor[0] - c_pts[0]), (coor[1] - c_pts[1])
        dis = np.sqrt((cha_x ** 2) + (cha_y ** 2)) * ratio
        ang = abs(np.arctan(cha_y / cha_x)) if cha_x != 0 else np.pi/2
        exp_coor = coor + (dis * np.cos(ang), dis * np.sin(ang))
        exp_coor = coor - (abs(dis*np.cos(ang)), abs(dis*np.sin(ang))
                           ) if cha_y < 0 and cha_x < 0 else exp_coor
        exp_coor = coor + (abs(dis*np.cos(ang)), -abs(dis*np.sin(ang))
                           ) if cha_y < 0 and cha_x > 0 else exp_coor
        exp_coor = coor + (-abs(dis*np.cos(ang)), abs(dis*np.sin(ang))
                           ) if cha_y >= 0 and cha_x <= 0 else exp_coor
        expand_poly[i] = exp_coor

    for x, y in expand_poly:
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        x = img_width - 1 if x >= img_width else x
        y = img_height - 1 if y >= img_height else y
    return expand_poly


def cal_Width_Height(Poly):
    (tl, tr, br, bl) = Poly
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    mean_Width = (int(widthA)+int(widthB))/2
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    mean_height = (int(heightA)+int(heightB))/2
    return maxWidth, maxHeight, mean_Width, mean_height


def four_point_transform(image, pts):
    quad = (pts)
    (tl, tr, br, bl) = quad
    maxWidth, maxHeight, mean_Width, mean_height = cal_Width_Height(quad)
    if mean_Width >= mean_height:
        Height = maxHeight
    else:
        quad = (tr, br, bl, tl)
        Height = maxWidth
    # expand quad
    if Height < 10:
        R = 1.0
    elif Height < 15:
        R = 0.8
    elif Height < 20:
        R = 0.3
    else:
        R = 0.2
    expand_quad = expand_roi(quad, ratio=R)
    maxWidth, maxHeight, _, _ = cal_Width_Height(expand_quad)
    rect = np.array([[0, 0], [maxWidth - 1, 0],
                     [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(expand_quad, rect)
    warped = cv2.warpPerspective(
        image, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)
    return warped, expand_quad, rect


def model_prediction(generator, device, input):
    H_, W_ = input.shape[0:2]
    text_region = Image.fromarray(cv2.cvtColor(
        input, cv2.COLOR_BGR2RGB))  # cvmat -> PILimage
    resize_H = H_ if H_ <= cfg.data_shape[0] else cfg.data_shape[0]
    resize_W = W_ if W_ <= cfg.data_shape[1] else cfg.data_shape[1]
    transform_list = [transforms.Resize((resize_H, resize_W), InterpolationMode.BICUBIC), transforms.ToTensor(
    ), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)
    input = transform(text_region)  # to tensor
    input_padding = input.new_full(
        (3, cfg.data_shape[0], cfg.data_shape[1]), 0)  # 128, 640
    input_padding[:input.shape[0], :input.shape[1],
                  :input.shape[2]].copy_(input)
    input_padding = input_padding.unsqueeze(0).to(device)
    with torch.no_grad():
        output_padding, o_mask_padding = generator(input_padding)
        output = output_padding[:, :, :input.shape[1], :input.shape[2]]
        o_mask = o_mask_padding[:, :, :input.shape[1], :input.shape[2]]
        display = torch.cat(((input.unsqueeze(0).clone(
        )*0.5+0.5).to(device), o_mask, (output.clone()*0.5+0.5)), 2)
        o_mask = tensor2jpg_mask(o_mask, W_, H_)
        display = tensor2jpg_mask(display, W_, 3*H_)
        output = tensor2jpg(output, W_, H_)
    return output, o_mask, display


def comp_back_persp(dst, output, img_height, img_width, rect, expand_poly, poly_pts):

    MM = cv2.getPerspectiveTransform(rect, expand_poly)
    text_erased_patch = cv2.warpPerspective(
        output, MM, (img_width, img_height), flags=cv2.INTER_CUBIC)
    mask = np.zeros((img_height, img_width, 1), np.uint8)
    poly_pts = poly_pts.astype(int).reshape((-1, 1, 2))
    cv2.fillConvexPoly(mask, poly_pts, (255, 255, 255))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3))  # cv2.MORPH_RECT
    mask = cv2.dilate(mask, kernel)
    mask = mask.reshape((img_height, img_width, -1))
    dst = np.where(mask > 0, text_erased_patch, dst)

    return dst


def inference(generator, device, src_img_dir, src_txt_dir, save_path):
    generator.eval()
    txt_list = [x for x in os.listdir(src_txt_dir) if is_txt_file(x)]
    txt_list.sort(key=lambda f: int("".join(list(filter(str.isdigit, f)))))

    for i, gt_txt in enumerate(txt_list[:]):  # 773
        gt_lines_txt = open(os.path.join(src_txt_dir, gt_txt), mode='r')
        gt_lines = gt_lines_txt.readlines()
        img_idx = gt_txt.replace('.txt', '').replace(
            'gt_', '').replace('res_', '')
        print(f'{i}/{len(txt_list)}', img_idx, end='\r')
        try:
            img = cv2.imread(os.path.join(src_img_dir, img_idx+'.jpg'))
            img_clone = img.copy()
        except Exception:
            img = cv2.imread(os.path.join(src_img_dir, img_idx+'.png'))
        img_clone = img.copy()
        dst = img.copy()
        mask_img = img.copy()
        img_height, img_width = img.shape[0:2]
        k = 1
        for gt in gt_lines:  # gt为单行标注
            line_parts = gt.strip().split(",")
            pts_num = int(len(line_parts)/2)
            poly = list(map(int, list(map(float, line_parts[0:pts_num*2]))))
            text = list(map(str, line_parts[pts_num*2: len(line_parts)]))
            poly_pts = np.array(poly, np.int32)
            if pts_num == 4:
                four_pts = poly_pts.reshape((-1, 2)).astype(float)
                input, expand_poly, rect = four_point_transform(
                    dst, four_pts)
                output, o_mask, display = model_prediction(
                    generator, device, input)
                dst = comp_back_persp(
                    dst, output, img_height, img_width, rect, expand_poly, poly_pts)
                mask_img = comp_back_persp(
                    mask_img, o_mask, img_height, img_width, rect, expand_poly, poly_pts)
                expand_poly = expand_poly.astype(int).reshape((-1, 1, 2))
            else:
                print('NOT RIGHT BBOX')

#             cv2.namedWindow('img_clone', cv2.WINDOW_NORMAL)
#             cv2.imshow('img_clone', img_clone)
#             cv2.imshow('dst', dst)
#             cv2.imshow('mask_img', mask_img)
#             windows_name = f'part_{k}'
#             cv2.imshow(windows_name, display)
#         k += 1
#         cv2.waitKey()
#         for ii in range(k):
#             cv2.destroyWindow(f'part_{ii}')                
                
        # save image
        cv2.imwrite(os.path.join(save_path, img_idx+'.png'), dst)




def quality_metric(img_path, label_path):
    img_list = [x for x in os.listdir(img_path) if is_image_file(x)]
    img_list.sort(key=lambda f: int("".join(list(filter(str.isdigit, f)))))
    psnr_list = []
    ssim_list = []
    mse_list = []
    for i, img_name in enumerate(img_list[:]):
        img = cv2.imread(os.path.join(img_path, img_name))
        try:
            label = cv2.imread(os.path.join(
                label_path, img_name.replace('.png', '.jpg')))
            label_clone = label.copy()
        except Exception:
            label = cv2.imread(os.path.join(
                label_path, img_name))
        psnr = cal_psnr(img, label)
        ssim = cal_ssim(img, label)
        mse = cal_mse(img, label)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        mse_list.append(mse)

    ave_psnr = np.mean(psnr_list)
    ave_ssim = np.mean(ssim_list)
    ave_mse = np.mean(mse_list)

    return ave_psnr, ave_ssim, ave_mse
