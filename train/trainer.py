import os
import torch
import numpy as np
from tensorboardX import SummaryWriter
from engine import train_one_epoch, evaluate
import sys
sys.path.append("..")
# TODO: improve metrics
from modules.utils import train_utils
from modules.utils import metrics


class Trainer(object):

    def __init__(self,
                 config,
                 model, 
                 criterion, 
                 optimizer, 
                 train_dataloader, 
                 valid_dataloader,
                 logger,
                 device,
                 activation_func=None,
                 USE_TENSORBOARD=True,
                 USE_CUDA=True,
                 ):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.logger = logger
        self.iterations = 0
        self.device = device
        self.activation_func = activation_func
        self.USE_TENSORBOARD = USE_TENSORBOARD
        self.checkpoint_path = self.config.CHECKPOINT_PATH
        if self.USE_TENSORBOARD:
            self.train_writer = SummaryWriter(log_dir=os.path.join(self.checkpoint_path, 'train'))
            self.valid_writer = SummaryWriter(log_dir=os.path.join(self.checkpoint_path, 'valid'))
        if USE_CUDA:
            self.model.cuda()
        self.max_acc = 0

    def fit(self):
        for self.epoch in range(1, self.config.TRAIN.EPOCH + 1):
            self.train()
            with torch.no_grad():
                self.validate()
                self.save_model()

        if self.USE_TENSORBOARD:
            self.train_writer.close()
            self.valid_writer.close()
            
    


    def train(self):
        for epoch in range(self.epoch):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.model, self.optimizer, self.train_dataloader, self.device, self.epoch, print_freq=10)
            # update the learning rate
            # lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(self.model, self.valid_dataloader, device=self.device)
        
        self.model.train()
        print(60*"=")
        self.logger.info(f'Epoch {self.epoch}/{self.config.TRAIN.EPOCH}')
        train_samples = len(self.train_dataloader.dataset)
        for i, data in enumerate(self.train_dataloader, self.iterations + 1):
            total_train_loss = 0.0
            # batch_input, batch_target = data
            # input_var = batch_input
            # target_var = batch_target
            # input_var, target_var = data['input'], data['gt']
            input_var, target_var = data
            input_var = train_utils.minmax_norm(input_var)
            input_var, target_var = input_var.to(self.device), target_var.to(self.device)

            # def closure():
            batch_output = self.model(input_var)
            
            import matplotlib.pyplot as plt
            # batch_output = self.activation_func(batch_output)
            if i%20 == 0:
                fig, ax = plt.subplots(1, 3)
                ax[0].imshow(input_var.cpu().detach().numpy()[0,0], 'gray')
                ax[0].imshow(target_var.cpu().detach().numpy()[0,0], cmap='jet', alpha=0.2)
                ax[1].imshow(batch_output.cpu().detach().numpy()[0,0], 'gray')
                ax[2].imshow(target_var.cpu().detach().numpy()[0,0], 'gray')
                # plt.show()
                fig.savefig(f'{i:03d}.png')

            # if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            #     loss = self.criterion(batch_output, torch.argmax(target_var.long(), axis=1))
            # else:
            #     loss = self.criterion(batch_output, target_var)
            loss = self.criterion(batch_output, target_var)
            loss.backward()
                # return loss
            self.optimizer.zero_grad()
            self.optimizer.step()
        
            loss = loss.item()
            total_train_loss += loss
            if self.USE_TENSORBOARD:
                self.train_writer.add_scalar('Loss/step', loss, i)

            display_step = train_utils.calculate_display_step(num_sample=train_samples, batch_size=self.config.DATA.BATCH_SIZE)
            # TODO: display_step = 10
            display_step = 10
            if i%display_step == 0:
                self.logger.info('Step {}  Step loss {}'.format(i, loss))
        self.iterations = i
        if self.USE_TENSORBOARD:
            self.train_writer.add_scalar('Loss/epoch', total_train_loss/train_samples, self.epoch)

    # def train(self):
    #     self.model.train()
    #     print(60*"=")
    #     self.logger.info(f'Epoch {self.epoch}/{self.config.TRAIN.EPOCH}')
    #     train_samples = len(self.train_dataloader.dataset)
    #     for i, data in enumerate(self.train_dataloader, self.iterations + 1):
    #         total_train_loss = 0.0
    #         # batch_input, batch_target = data
    #         # input_var = batch_input
    #         # target_var = batch_target
    #         # input_var, target_var = data['input'], data['gt']
    #         input_var, target_var = data
    #         input_var = train_utils.minmax_norm(input_var)
    #         input_var, target_var = input_var.to(self.device), target_var.to(self.device)

    #         # def closure():
    #         batch_output = self.model(input_var)
            
    #         import matplotlib.pyplot as plt
    #         # batch_output = self.activation_func(batch_output)
    #         if i%20 == 0:
    #             fig, ax = plt.subplots(1, 3)
    #             ax[0].imshow(input_var.cpu().detach().numpy()[0,0], 'gray')
    #             ax[0].imshow(target_var.cpu().detach().numpy()[0,0], cmap='jet', alpha=0.2)
    #             ax[1].imshow(batch_output.cpu().detach().numpy()[0,0], 'gray')
    #             ax[2].imshow(target_var.cpu().detach().numpy()[0,0], 'gray')
    #             # plt.show()
    #             fig.savefig(f'{i:03d}.png')

    #         # if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
    #         #     loss = self.criterion(batch_output, torch.argmax(target_var.long(), axis=1))
    #         # else:
    #         #     loss = self.criterion(batch_output, target_var)
    #         loss = self.criterion(batch_output, target_var)
    #         loss.backward()
    #             # return loss
    #         self.optimizer.zero_grad()
    #         self.optimizer.step()
        
    #         loss = loss.item()
    #         total_train_loss += loss
    #         if self.USE_TENSORBOARD:
    #             self.train_writer.add_scalar('Loss/step', loss, i)

    #         display_step = train_utils.calculate_display_step(num_sample=train_samples, batch_size=self.config.DATA.BATCH_SIZE)
    #         # TODO: display_step = 10
    #         display_step = 10
    #         if i%display_step == 0:
    #             self.logger.info('Step {}  Step loss {}'.format(i, loss))
    #     self.iterations = i
    #     if self.USE_TENSORBOARD:
    #         self.train_writer.add_scalar('Loss/epoch', total_train_loss/train_samples, self.epoch)

    def validate(self):
        self.model.eval()
        self.eval_tool = metrics.SegmentationMetrics(self.config.MODEL.NUM_CLASSES, ['accuracy'])
        test_n_iter, total_test_loss = 0, 0
        valid_samples = len(self.valid_dataloader.dataset)
        for _, data in enumerate(self.valid_dataloader):
            test_n_iter += 1
            # inputs, labels = data['input'], data['gt']
            inputs, labels = data
            inputs = train_utils.minmax_norm(inputs)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)

            # if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            #     loss = self.criterion(outputs, torch.argmax(labels.long(), axis=1))
            # else:
            #     loss = self.criterion(outputs, labels)
            loss = self.criterion(outputs, labels)

            # loss = loss_func(outputs, torch.argmax(labels, dim=1)).item()
            total_test_loss += loss.item()

            # TODO: torch.nn.functional.sigmoid(outputs)
            # prob = torch.nn.functional.softmax(outputs, dim=1)
            # prob = torch.sigmoid(outputs)
            if self.activation_func:
                prob = self.activation_func(outputs)
            else:
                prob = outputs
            prediction = torch.argmax(prob, dim=1)
            labels = torch.argmax(labels, dim=1)
            labels = labels.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            evals = self.eval_tool(labels, prediction)

        self.avg_test_acc = metrics.accuracy(
                np.sum(self.eval_tool.total_tp), np.sum(self.eval_tool.total_fp), np.sum(self.eval_tool.total_fn), np.sum(self.eval_tool.total_tn)).item()
        self.valid_writer.add_scalar('Accuracy/epoch', self.avg_test_acc, self.epoch)
        self.valid_writer.add_scalar('Loss/epoch', total_test_loss/valid_samples, self.epoch)

    def load_model_from_checkpoint():
        pass

    def save_model(self):
        checkpoint = {
            "net": self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "epoch": self.epoch
            }
        if self.avg_test_acc > self.max_acc:
            self.max_acc = self.avg_test_acc
            self.logger.info(f"-- Saving best model with testing accuracy {self.max_acc:.3f} --")
            checkpoint_name = 'ckpt_best.pth'
            torch.save(checkpoint, os.path.join(self.checkpoint_path, checkpoint_name))

        if self.epoch%self.config.TRAIN.CHECKPOINT_SAVING_STEPS == 0:
            self.logger.info(f"Saving model with testing accuracy {self.avg_test_acc:.3f} in epoch {self.epoch} ")
            checkpoint_name = 'ckpt_best_{:04d}.pth'.format(self.epoch)
            torch.save(checkpoint, os.path.join(self.checkpoint_path, checkpoint_name))



if __name__ == '__main__':
    pass