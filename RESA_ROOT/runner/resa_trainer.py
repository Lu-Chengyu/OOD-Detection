import torch.nn as nn
import torch
import torch.nn.functional as F
from .Myloss import L_color, L_spa, L_exp, L_TV
from runner.registry import TRAINER

def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    return (1-d).mean()

@TRAINER.register_module
class RESA(nn.Module):
    def __init__(self, cfg):
        super(RESA, self).__init__()
        self.cfg = cfg
        self.loss_type = cfg.loss_type
        if self.loss_type == 'cross_entropy':
            weights = torch.ones(cfg.num_classes)
            weights[0] = cfg.bg_weight
            weights = weights.cuda()
            self.criterion = torch.nn.NLLLoss(ignore_index=self.cfg.ignore_label,
                                              weight=weights).cuda()

        self.criterion_exist = torch.nn.BCEWithLogitsLoss().cuda()
        self.L_color = L_color()
        self.L_spa = L_spa()

        self.L_exp = L_exp(16, 0.5)
        self.L_TV = L_TV()

    def forward(self, net, batch, mean):  # gai
        output = net(batch['img'], mean)  # gai

        loss_stats = {}
        loss = 0.

        # gaikai


        Loss_TV = 200 * self.L_TV(output['A'])

        loss_spa = torch.mean(self.L_spa(output['enhanced_image'], output['img_lowlight']))

        loss_col = 5 * torch.mean(self.L_color(output['enhanced_image']))

        loss_exp = 10 * torch.mean(self.L_exp(output['enhanced_image']))

        # best_loss
        loss_enhence = Loss_TV + loss_spa + loss_col + loss_exp
        # gaijie

        if self.loss_type == 'dice_loss':
            target = F.one_hot(batch['label'], num_classes=self.cfg.num_classes).permute(0, 3, 1, 2)
            seg_loss = dice_loss(F.softmax(
                output['seg'], dim=1)[:, 1:], target[:, 1:])
        else:
            seg_loss = self.criterion(F.log_softmax(
                output['seg'], dim=1), batch['label'].long())

        loss += seg_loss * self.cfg.seg_loss_weight

        loss_stats.update({'seg_loss': seg_loss})

        if 'exist' in output:
            exist_loss = 0.1 * \
                self.criterion_exist(output['exist'], batch['exist'].float())
            loss += exist_loss
            loss_stats.update({'exist_loss': exist_loss})
        loss_en = loss_enhence * 0.1
        loss_stats.update({'loss_enhence': loss_en})
        loss = loss + loss_en  # 改

        ret = {'loss': loss, 'loss_stats': loss_stats }

        return ret
