import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import datasets
import os.path as osp
from torch.autograd import Variable
from models import ImgNet, TxtNet, GCN_txt, GCN_img
from utils import compress_wiki, compress, calculate_top_map,logger


class JAGH:
    def __init__(self):
        self.logger = logger()

        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.cuda.set_device(config.GPU_ID)

        if config.DATASET == "WIKI":
            self.train_dataset = datasets.WIKI(root=config.DATA_DIR, train=True, transform=datasets.wiki_train_transform)
            self.test_dataset = datasets.WIKI(root=config.DATA_DIR, train=False, transform=datasets.wiki_test_transform)
            self.database_dataset = datasets.WIKI(root=config.DATA_DIR, train=True, transform=datasets.wiki_test_transform)

        if config.DATASET == "MIRFlickr":
          self.train_dataset = datasets.MIRFlickr(train=True, transform=datasets.mir_train_transform)
          self.test_dataset = datasets.MIRFlickr(train=False, database=False, transform=datasets.mir_test_transform)
          self.database_dataset = datasets.MIRFlickr(train=False, database=True, transform=datasets.mir_test_transform)

        if config.DATASET == "NUSWIDE":
            self.train_dataset = datasets.NUSWIDE(train=True, transform=datasets.nus_train_transform)
            self.test_dataset = datasets.NUSWIDE(train=False, database=False, transform=datasets.nus_test_transform)
            self.database_dataset = datasets.NUSWIDE(train=False, database=True, transform=datasets.nus_test_transform)

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=config.BATCH_SIZE,
                                                        shuffle=True,
                                                        num_workers=config.NUM_WORKERS,
                                                        drop_last=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=config.BATCH_SIZE,
                                                       shuffle=False,
                                                       num_workers=config.NUM_WORKERS)

        self.database_loader = torch.utils.data.DataLoader(dataset=self.database_dataset,
                                                           batch_size=config.BATCH_SIZE,
                                                           shuffle=False,
                                                           num_workers=config.NUM_WORKERS)

        self.CodeNet_I = ImgNet(code_len=config.HASH_BIT)
        self.FeatNet_I = ImgNet(code_len=config.HASH_BIT)
        self.GCN_I = GCN_img(code_len=config.HASH_BIT, gamma=config.gamma)
        self.GCN_T = GCN_txt(code_len=config.HASH_BIT, gamma=config.gamma)

        txt_feat_len = datasets.txt_feat_len
        self.CodeNet_T = TxtNet(code_len=config.HASH_BIT, txt_feat_len=txt_feat_len)

        if config.DATASET == "WIKI":
            self.opt_I = torch.optim.SGD([{'params': self.CodeNet_I.fc_encode.parameters(), 'lr': config.LR_IMG},
                                          {'params': self.CodeNet_I.alexnet.classifier.parameters(), 'lr': config.LR_IMG}],
                                         momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)

        if config.DATASET == "MIRFlickr" or config.DATASET == "NUSWIDE":
            self.opt_I = torch.optim.SGD(self.CodeNet_I.parameters(), lr=config.LR_IMG, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)

        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(), lr=config.LR_TXT, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)

        self.opt_GCN_I = torch.optim.Adam(self.GCN_I.parameters(), lr=0.0001)
        #self.opt_GCN_I = torch.optim.SGD(self.GCN_I.parameters(), lr=0.001, momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)
        self.opt_GCN_T = torch.optim.Adam(self.GCN_T.parameters(), lr=0.0001)
        #self.opt_GCN_T = torch.optim.SGD(self.GCN_T.parameters(), lr=0.001, momentum=settings.MOMENTUM, weight_decay=settings.WEIGHT_DECAY)

        self.best_it = 0
        self.best_ti = 0

    def train(self, epoch):

        self.CodeNet_I.cuda().train()
        self.FeatNet_I.cuda().eval()
        self.CodeNet_T.cuda().train()

        self.GCN_I.cuda().train()
        self.GCN_T.cuda().train()

        self.CodeNet_I.set_alpha(epoch)
        self.CodeNet_T.set_alpha(epoch)
        self.GCN_I.set_alpha(epoch)
        self.GCN_T.set_alpha(epoch)

        self.logger.info('Epoch [%d/%d], alpha for ImgNet: %.3f, alpha for TxtNet: %.3f' % (epoch + 1, config.NUM_EPOCH, self.CodeNet_I.alpha, self.CodeNet_T.alpha))

        for idx, (img, txt, _, _) in enumerate(self.train_loader):

            img = Variable(img.cuda())
            txt = Variable(torch.FloatTensor(txt.numpy()).cuda())

            self.opt_I.zero_grad()
            self.opt_T.zero_grad()
            self.opt_GCN_I.zero_grad()
            self.opt_GCN_T.zero_grad()

            F_I , _, _ = self.FeatNet_I(img)

            F_I = F.normalize(txt)
            S_I = F_I.mm(F_I.t())
            S_I = S_I * 2 - 1

            F_T = F.normalize(F_T)
            S_T = F_T.mm(F_T.t())
            S_T = S_T * 2 - 1

            S = (config.alpha * S_I + config.beta * S_T + config.lamb * F.normalize(S_I.mm(S_T))) * config.mu

            fea_I, hid_I, code_I = self.CodeNet_I(img)
            fea_T, hid_T, code_T = self.CodeNet_T(txt)

            code_gcn_I = self.GCN_I(fea_I.detach(), S)
            code_gcn_T = self.GCN_T(fea_T.detach(), S)

            B_I = F.normalize(code_I)
            B_T = F.normalize(code_T)
            B_gI = F.normalize(code_gcn_I)
            B_gT = F.normalize(code_gcn_T)

            BI_BI = B_I.mm(B_I.t())
            BT_BT = B_T.mm(B_T.t())
            BI_BT = B_I.mm(B_T.t())
            GBI_GBT = B_gI.mm(B_gT.t())
            GI = B_gI.mm(B_I.t())
            GT = B_gI.mm(B_T.t())

            loss1 = F.mse_loss(BI_BI, S)
            loss2 = F.mse_loss(BI_BT, S)
            loss3 = F.mse_loss(BT_BT, S)
            loss4 = F.mse_loss(GBI_GBT, S)
            loss5 = F.mse_loss(S, GI) + F.mse_loss(S, GT)

            loss = config.INTRA * loss1 + loss2 + config.INTRA * loss3 + loss4 + config.pho3 * loss5

            loss.backward()
            self.opt_I.step()
            self.opt_T.step()
            self.opt_GCN_I.step()
            self.opt_GCN_T.step()

            if (idx + 1) % (len(self.train_dataset) // config.BATCH_SIZE / config.EPOCH_INTERVAL) == 0:
                self.logger.info('Epoch [%d/%d], Iter [%d/%d] Loss1: %.4f Loss2: %.4f Loss3: %.4f Total Loss: %.4f'
                    % (epoch + 1, config.NUM_EPOCH, idx + 1, len(self.train_dataset) // config.BATCH_SIZE,
                       loss1.item(), loss2.item(), loss3.item(), loss.item()))

    def eval(self):
        self.logger.info('--------------------Evaluation: mAP@50-------------------')

        self.CodeNet_I.eval().cuda()
        self.CodeNet_T.eval().cuda()

        if config.DATASET == "WIKI":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress_wiki(self.database_loader, self.test_loader, self.CodeNet_I, self.CodeNet_T, self.database_dataset, self.test_dataset)
        
        if config.DATASET == "MIRFlickr" or config.DATASET == "NUSWIDE":
            re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.CodeNet_I, self.CodeNet_T, self.database_dataset, self.test_dataset)
          
        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)

        if (self.best_it + self.best_ti) < (MAP_I2T+ MAP_T2I):
            self.best_it = MAP_I2T
            self.best_ti = MAP_T2I
        self.logger.info('mAP@50 I->T: %.3f, mAP@50 T->I: %.3f' % (MAP_I2T, MAP_T2I))
        self.logger.info('Best MAP of I->T: %.3f, Best mAP of T->I: %.3f' % (self.best_it, self.best_ti))
        self.logger.info('--------------------------------------------------------------------')


    def save_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(config.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.CodeNet_I.state_dict(),
            'TxtNet': self.CodeNet_T.state_dict(),
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')

    
    def load_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(config.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** No checkpoint %s!*********' % ckp_path)
            return
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])

    def log(self):

        self.logger.info('--- Configs List---')
        self.logger.info('--- Dadaset:{}'.format(config.DATASET))
        self.logger.info('--- Bit:{}'.format(config.CODE_LEN))
        self.logger.info('--- Alpha:{}'.format(config.alpha))
        self.logger.info('--- Beta:{}'.format(config.beta))
        self.logger.info('--- Lambda:{}'.format(config.lamb))
        self.logger.info('--- Gamma:{}'.format(config.gamma))
        self.logger.info('--- Pho:{}'.format(config.pho3))
        self.logger.info('--- Mu:{}'.format(config.mu))
        self.logger.info('--- Batch:{}'.format(config.BATCH_SIZE))




