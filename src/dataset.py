from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import cfg


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(DatasetFromFolder, self).__init__()
        self.i_s_path = join(dataset_dir, "i_s")
        self.t_b_path = join(dataset_dir, "t_b")
        self.mask_t_path = join(dataset_dir, "mask_t")
        self.image_filenames = [x for x in listdir(
            self.i_s_path) if is_image_file(x)]

    def __getitem__(self, index):
        i_s = Image.open(
            join(self.i_s_path, self.image_filenames[index])).convert('RGB')
        W, H = i_s.size
        t_b = Image.open(
            join(self.t_b_path, self.image_filenames[index])).convert('RGB')
        mask_t = Image.open(join(self.mask_t_path, self.image_filenames[index].replace(
            '.jpg', '.png'))).convert('RGB')
        resize_H = H if H <= cfg.data_shape[0] else cfg.data_shape[0]
        resize_W = W if W <= cfg.data_shape[1] else cfg.data_shape[1]
        transform_list = [transforms.Resize((resize_H, resize_W), InterpolationMode.BICUBIC),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform = transforms.Compose(transform_list)
        transform_list_1 = [transforms.Resize(
            (resize_H, resize_W), InterpolationMode.BICUBIC), transforms.ToTensor()]
        transform_1 = transforms.Compose(transform_list_1)
        i_s = transform(i_s)
        t_b = transform(t_b)
        mask_t = transform_1(mask_t)
        mask_t = 1. - mask_t

        i_s_padding = i_s.new_full(
            (3, cfg.data_shape[0], cfg.data_shape[1]), 0)
        i_s_padding[:i_s.shape[0], : i_s.shape[1], : i_s.shape[2]].copy_(i_s)
        t_b_padding = i_s.new_full(
            (3, cfg.data_shape[0], cfg.data_shape[1]), 0)
        t_b_padding[:t_b.shape[0], : t_b.shape[1], : t_b.shape[2]].copy_(t_b)
        mask_t_padding = i_s.new_full(
            (3, cfg.data_shape[0], cfg.data_shape[1]), 1)
        mask_t_padding[:mask_t.shape[0], : mask_t.shape[1],
                       : mask_t.shape[2]].copy_(mask_t)

        return i_s_padding, mask_t_padding, t_b_padding

    def __len__(self):
        return len(self.image_filenames)
