import os
from src.evaluate import quality_metric


def main(result_path, label_path):

    ave_psnr, ave_ssim, ave_mse = quality_metric(
        img_path=result_path, label_path=label_path)

    print(
        f'average psnr: {ave_psnr:.4f}, average ssim: {ave_ssim:.4f}, mse: {ave_mse:.8f}')


if __name__ == '__main__':
    result_path = "./_SCUT-EnsText_result"
    label_path = './SCUT-EnsText/test_set/all_labels'

    main(result_path, label_path)
