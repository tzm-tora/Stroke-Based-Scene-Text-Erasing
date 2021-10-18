import torch
import os
from src.evaluate import inference
from src.model import build_generator
from src.utils import makedirs, fix_model_state_dict
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(model_path, src_img_dir, src_txt_dir, save_path):
    # load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, "Loading the Model...")
    generator = build_generator()
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(fix_model_state_dict(checkpoint['net_G']))
    generator.to(device)

    makedirs(save_path)

    inference(generator, device, src_img_dir,
              src_txt_dir, save_path)


if __name__ == '__main__':

    model_path = './best.pth'
    src_img_dir = "./example/images"
    src_txt_dir = "./example/txts"

    save_path = f"./example/_inference_result"

    main(model_path, src_img_dir, src_txt_dir, save_path)
