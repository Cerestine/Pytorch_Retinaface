"""Module to train RetinaFace"""
import math
import time
import datetime
import torch
from torch.backends import cudnn
from torch import optim
from torch.utils import data
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from data import detection_collate

class RetinaFaceTrainer():
    """RetinaFace trainer class"""
    def __init__(self, net, model_params, config_data):
        self.net = net
        self.config_data = config_data
        cudnn.benchmark = True
        self.optimizer = optim.SGD(
            net.parameters(),
            lr=model_params.initial_lr,
            momentum=model_params.momentum,
            weight_decay=model_params.weight_decay)
        self.criterion = MultiBoxLoss(
            model_params.num_classes, 0.35, True, 0, True, 7, 0.35, False)
        self.priorbox = PriorBox(self.config_data,
                                 image_size=(self.config_data["img_dim"], self.config_data["img_dim"]))
        with torch.no_grad():
            self.priors = self.priorbox.forward()
            self.priors = self.priors.cuda()

    def adjust_learning_rate(self, optimizer, gamma, epoch, step_index, iteration, epoch_size):
        """Sets the learning rate
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        warmup_epoch = -1
        if epoch <= warmup_epoch:
            lr = 1e-6 + (self.config_data["initial_lr"]-1e-6) * iteration / (epoch_size * warmup_epoch)
        else:
            lr = self.config_data["initial_lr"] * (gamma ** (step_index))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train(self, resume_epoch, training_dataset):
        """Actual train runner"""
        self.net.train()
        epoch = 0 + resume_epoch
        epoch_size = math.ceil(len(training_dataset) / self.config_data["batch_size"])
        max_iter = self.config_data["max_epoch"] * epoch_size

        stepvalues = (
            self.config_data["step_dacay"]["decay_1"] * epoch_size, 
            self.config_data["step_dacay"]["decay_2"] * epoch_size)
        step_index = 0

        # if resume_epoch > 0:
        #     start_iter = resume_epoch * epoch_size
        # else:
        #     start_iter = 0

        start_iter = resume_epoch * epoch_size if resume_epoch > 0 else 0

        for iteration in range(start_iter, max_iter):
            if iteration % epoch_size == 0:
                # create batch iterator
                batch_iterator = iter(data.DataLoader(
                    training_dataset, self.config_data["batch_size"], shuffle=True, 
                    num_workers=self.config_data["num_workers"], collate_fn=detection_collate))
                if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > self.config_data["step_dacay"]["decay_1"]):
                    torch.save(self.net.state_dict(),
                               self.config_data["save_folder"] + self.config_data["name"]+ '_epoch_' + str(epoch) + '.pth')
                epoch += 1

            load_t0 = time.time()
            if iteration in stepvalues:
                step_index += 1
            lr = self.adjust_learning_rate(self.optimizer, self.config_data["gamma"],
                                            epoch, step_index, iteration, epoch_size)

            # load train data
            images, targets = next(batch_iterator)
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]

            # forward
            out = self.net(images)

            # backprop
            self.optimizer.zero_grad()
            loss_l, loss_c, loss_landm = self.criterion(out, self.priors, targets)
            loss = self.config_data["loc_weight"] * loss_l + loss_c + loss_landm
            loss.backward()
            self.optimizer.step()
            load_t1 = time.time()
            batch_time = load_t1 - load_t0
            eta = int(batch_time * (max_iter - iteration))
            print("Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{}\n".format(
                epoch, self.config_data["max_epoch"],
                (iteration % epoch_size) + 1, epoch_size,
                iteration + 1, max_iter))
            print("Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f}\nBatchtime: {:.4f} s || ETA: {}".format(
                loss_l.item(), loss_c.item(), loss_landm.item(), lr,
                batch_time, str(datetime.timedelta(seconds=eta))))

        torch.save(self.net.state_dict(), self.config_data["save_folder"] + self.config_data["name"] + '_Final.pth')
        # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')
