import time
import torch
import numpy as np
from tqdm import tqdm
import pytorch_warmup as warmup

from models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from datasets import build_dataloader
from .recorder import build_recorder
from .net_utils import save_model, load_network
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.preprocessing import StandardScaler

class Runner(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = range(self.cfg.gpus)).cuda()
        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.evaluator = build_evaluator(self.cfg)
        self.warmup_scheduler = warmup.LinearWarmup(
            self.optimizer, warmup_period=5000)
        self.metric = 0.
        self.mean = torch.ones(1, 1, 288, 800) # gai

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from,
                finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            batch[k] = batch[k].cuda()
        return batch
    
    def train_epoch(self, epoch, train_loader):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - end
            self.recorder.step += 1
            data = self.to_cuda(data)
            output = self.trainer.forward(self.net, data, self.mean)  # gai
            self.optimizer.zero_grad()
            loss = output['loss']
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.warmup_scheduler.dampen()
            batch_time = time.time() - end
            end = time.time()
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.lr = lr
                self.recorder.record('train')

    def train(self):
        self.recorder.logger.info('start training...')
        self.trainer = build_trainer(self.cfg)
        train_loader = build_dataloader(self.cfg.dataset.train, self.cfg, is_train=True)
        val_loader = build_dataloader(self.cfg.dataset.val, self.cfg, is_train=False)
        self.mean.cuda()  # gai
        for epoch in range(self.cfg.epochs):
            self.recorder.epoch = epoch
            self.train_epoch(epoch, train_loader)
            if (epoch + 1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt()
            if (epoch + 1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate(val_loader)
            if self.recorder.step >= self.cfg.total_iter:
                break

    def validate(self, val_loader):
        self.net.eval()
        self.mean.cuda()
        for i, data in enumerate(tqdm(val_loader, desc=f'Validate')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data['img'], self.mean)
                self.evaluator.evaluate(val_loader.dataset, output, data)

        metric = self.evaluator.summarize()
        if not metric:
            return
        if metric > self.metric:
            self.metric = metric
            self.save_ckpt(is_best=True)
        self.recorder.logger.info('Best metric: ' + str(self.metric))

    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler,
                self.recorder, is_best)
        

    def pca(self):  # gai
        pca_train_loader = build_dataloader(self.cfg.dataset.train, self.cfg, is_train=False) # 防止數據增廣
        pca_test_loader = build_dataloader(self.cfg.dataset.val, self.cfg, is_train=False)
        self.net.eval()
        self.mean.cuda()

        cache_dir = '/fs/scratch/sgh_cr_bcai_dl_cluster_users/archive/CULaneFeature/CULane' # /CULane
        train_feat = np.memmap(f"{cache_dir}/trainFeature.mmap", dtype=np.float32, mode='w+', shape=(len(pca_train_loader.dataset), 460800))
        test_feat = np.memmap(f"{cache_dir}/testFeature.mmap", dtype=np.float32, mode='w+', shape=(len(pca_test_loader.dataset), 460800))

        for i, data in enumerate(tqdm(pca_train_loader, desc=f'PcaTrain')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data['img'], self.mean)
                seg, laneFeature = output['seg'], output['lane_feature'] # gai
                laneFeature = laneFeature.cpu().numpy() # gai
                batch_size = seg.size(0)
                laneFeature = laneFeature.reshape(batch_size, -1)
                start_ind = i * batch_size
                end_ind = min((i + 1) * batch_size, len(pca_train_loader.dataset))
                train_feat[start_ind:end_ind, :] = laneFeature

        for i, data in enumerate(tqdm(pca_test_loader, desc=f'PcaTestSet')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data['img'], self.mean)
                seg, laneFeature = output['seg'], output['lane_feature'] # gai
                laneFeature = laneFeature.cpu().numpy() # gai
                batch_size = seg.size(0)
                laneFeature = laneFeature.reshape(batch_size, -1)
                start_ind = i * batch_size
                end_ind = min((i + 1) * batch_size, len(pca_test_loader.dataset))
                test_feat[start_ind:end_ind, :] = laneFeature