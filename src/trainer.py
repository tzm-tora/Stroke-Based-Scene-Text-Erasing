import torch
import cfg
from tqdm import tqdm
from torch.utils.data import DataLoader
from .utils import to_items, reduce_value
from torchvision.utils import make_grid, save_image
from .utils import create_ckpt_dir, makedirs
from torch.utils.tensorboard import SummaryWriter
from .model import build_generator
from src.dataset import DatasetFromFolder
from src.loss import InpaintingLoss


class Trainer(object):
    def __init__(self, device, rank):
        self.flag_epoch = 0
        self.global_step = 0
        self.device = device
        self.rank = rank

        self.train_loader = self.make_data_loader()
        self.net_G = self.init_net(build_generator(
            finetune=cfg.finetune).to(self.device))

        self.optimizer_G, self.lr_scheduler_G = self.init_optimizer()
        self.Inpainting_loss = InpaintingLoss().to(self.device)

        if cfg.resume:
            self.resume_model()
        if cfg.finetune:
            self.finetune_model()

        self.loss_dict = {}
        self.tb_writer = None
        if self.rank == 0:
            self.save_dir, self.tensorboard_dir = create_ckpt_dir(cfg.ckpt_dir)
            self.tb_writer = SummaryWriter(self.tensorboard_dir)

    def iterate(self):
        if self.rank == 0:
            print('Start the training')
        for epoch in range(self.flag_epoch, cfg.max_epoch + 1):
            if self.rank == 0:
                print(f'[epoch {epoch:>3}]')
                self.train_loader = tqdm(self.train_loader)
            self.train_sampler.set_epoch(epoch)
            for step, (input, t_mask, gt) in enumerate(self.train_loader):
                self.global_step += 1
                input, t_mask, gt = input.to(self.device), t_mask.to(
                    self.device), gt.to(self.device)
                output, o_mask = self.train_step(input, t_mask, gt)

                # log the loss and img
                if self.rank == 0 and self.global_step % (cfg.log_interval) == 0:
                    self.log_loss()
                if self.rank == 0 and self.global_step % (cfg.log_interval*5) == 0:
                    self.log_img(input, o_mask, t_mask, output, gt)

            self.lr_scheduler_G.step()

            # save the model
            if self.rank == 0 and epoch % cfg.save_model_interval == 0 and epoch >= 1:
                self.save_model(epoch)

    def train_step(self, input, t_mask, gt):
        self.net_G.train()
        output, o_mask = self.net_G(input)
        # update G
        self.optimizer_G.zero_grad()
        self.backward_G(input, o_mask, t_mask, output, gt)
        self.optimizer_G.step()
        return output, o_mask

    def backward_G(self, input, o_mask, t_mask, output, gt):
        Inpainting_loss, loss_dict = self.Inpainting_loss(
            input, o_mask, t_mask, output, gt)
        G_Loss = Inpainting_loss
        G_Loss.backward()
        loss_dict['generator'] = reduce_value(G_Loss, average=True)
        self.loss_dict.update(loss_dict)

    def init_net(self, net):
        if self.rank == 0:
            print(f"Loading the net in GPU")
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            net).to(self.device)
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[self.rank], find_unused_parameters=False)
        return net

    def init_optimizer(self):
        lr = cfg.finetune_lr if cfg.finetune else cfg.initial_lr
        optimizer_G = torch.optim.Adam(filter(
            lambda p: p.requires_grad, self.net_G.parameters()), lr=lr)
        lr_scheduler_G = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_G, gamma=0.95)
        return optimizer_G, lr_scheduler_G

    def make_data_loader(self):
        if self.rank == 0:
            print("Loading Dataset...")
        train_dataset = DatasetFromFolder(cfg.train_data_root)
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        train_batch_sampler = torch.utils.data.BatchSampler(
            self.train_sampler, cfg.batch_size, drop_last=True)
        train_loader = DataLoader(
            train_dataset, batch_sampler=train_batch_sampler, shuffle=False, num_workers=8, pin_memory=True)
        return train_loader

    def report(self, epoch, step, batch_num, loss_dict):
        print('[epoch {:>3}] | [STEP: {:>4}/{:>4d}] | Total Loss: {:.4f}'.format(
            epoch, step, batch_num, loss_dict['total']))

    def log_loss(self):
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(
                'detail_loss', self.loss_dict, self.global_step)
            self.tb_writer.add_scalar(
                'total_Loss/generator', self.loss_dict['generator'], self.global_step)
            self.tb_writer.add_scalar(
                'LR/lr', self.optimizer_G.state_dict()['param_groups'][0]['lr'], self.global_step)

    def log_img(self, input, o_mask, t_mask, output, gt):
        if self.tb_writer is not None:
            dis_row = 2  # <= batchsize
            output_comp = o_mask * input + (1. - o_mask) * output
            images = torch.cat((input[0:dis_row, ...], output[0:dis_row, ...],
                                output_comp[0:dis_row, ...], gt[0:dis_row, ...]), 0)
            images = images*0.5 + 0.5
            images = torch.cat(
                (images, o_mask[0:dis_row, ...], t_mask[0:dis_row, ...]), 0)
            grid = make_grid(images, nrow=dis_row,
                             padding=10)
            self.tb_writer.add_image('train', grid, self.global_step)

    def save_model(self, epoch, note=''):
        print('Saving the model...')
        save_files = {'net_G': self.net_G.module.state_dict(),
                      'optimizer_G': self.optimizer_G.state_dict(),
                      'lr_scheduler_G': self.lr_scheduler_G.state_dict(),
                      'epoch': epoch,
                      'global_step': self.global_step}
        torch.save(save_files, f'{self.save_dir}/{note}{epoch}.pth')

    def resume_model(self):
        print("Loading the trained params and the state of optimizer...")
        checkpoint = torch.load(cfg.resume_path, map_location=self.device)
        self.net_G.module.load_state_dict(checkpoint['net_G'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])
        self.flag_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        print(
            f"Resuming from epoch: {self.flag_epoch}, global step: {self.global_step}")

    def finetune_model(self):
        if cfg.finetune:
            checkpoint = torch.load(cfg.resume_path, map_location=self.device)
            self.net_G.module.load_state_dict(checkpoint['net_G'])
            flag_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']
            print(
                f"finetuning from epoch: {flag_epoch}, global step: {global_step}")
