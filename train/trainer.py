import os
import torch
import numpy as np
from tensorboardX import SummaryWriter
from utils import train_utils
# TODO: Try to Minimum libraries dependency


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
                 checkpoint_path,
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
        self.USE_TENSORBOARD = USE_TENSORBOARD
        self.checkpoint_path = checkpoint_path
        if self.USE_TENSORBOARD:
            self.train_writer = SummaryWriter(log_dir=os.path.join(self.checkpoint_path, 'train'))
            self.test_writer = SummaryWriter(log_dir=os.path.join(self.checkpoint_path, 'valid'))
        if USE_CUDA:
            self.model.cuda()

    def fit(self):
        for self.epoch in range(1, self.config.train.epoch + 1):
            self.train()
            with torch.no_grad():
                self.validate()
                self.save_model()

        if self.USE_TENSORBOARD:
            self.train_writer.close()
            self.test_writer.close()
            
    def train(self):
        self.model.train()
        print(60*"=")
        self.logger.info(f'Epoch {self.epoch}/{self.config.train.epoch}')
        train_samples = len(self.train_dataloader.dataset)
        for i, data in enumerate(self.train_dataloader, self.iterations + 1):
            total_train_loss = 0.0
            # batch_input, batch_target = data
            # input_var = batch_input
            # target_var = batch_target
            input_var, target_var = data['input'], data['gt']
            input_var = train_utils.minmax_norm(input_var)
            input_var = input_var.to(self.device)
            target_var = target_var.to(self.device)
            def closure():
                batch_output = self.model(input_var)
                if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                    loss = self.criterion(batch_output, torch.argmax(target_var.long(), axis=1))
                else:
                    loss = self.criterion(batch_output, target_var)
                # loss = self.criterion(batch_output, target_var)
                loss.backward()
                return loss
            self.optimizer.zero_grad()
            self.optimizer.step(closure)
        
            loss = loss.item()
            total_train_loss += loss
            if self.USE_TENSORBOARD:
                self.train_writer.add_scalar('Loss/step', loss, self.iterations)

            display_step = train_utils.calculate_display_step(num_sample=train_samples, batch_size=self.config.dataset.batch_size)
            if i%display_step == 0:
                self.logger.info('Step {}  Step loss {}'.format(i, loss))
        self.iterations += i
        if self.USE_TENSORBOARD:
            self.train_writer.add_scalar('Loss/epoch', total_train_loss/train_samples, self.epoch)

    def validate(self):
        self.model.eval()
        eval_tool = metrics.SegmentationMetrics(self.config.model.out_channels, ['accuracy'])
        test_n_iter, total_test_loss = 0, 0
        for _, data in enumerate(self.valid_dataloader):
            test_n_iter += 1
            inputs, labels = data['input'], data['gt']
            inputs = train_utils.minmax_norm(inputs)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                loss = self.criterion(outputs, torch.argmax(labels.long(), axis=1))
            else:
                loss = self.criterion(outputs, labels)
            # loss = loss_func(outputs, torch.argmax(labels, dim=1)).item()
            total_test_loss += loss.item()

            # TODO: torch.nn.functional.sigmoid(outputs)
            # prob = torch.nn.functional.softmax(outputs, dim=1)
            prob = torch.sigmoid(outputs)
            prediction = torch.argmax(prob, dim=1)
            labels = torch.argmax(labels, dim=1)
            labels = labels.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            evals = eval_tool(labels, prediction)

    def load_model_from_checkpoint():
        pass

    def save_model(self):
        checkpoint = {
            "net": self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "epoch": self.epoch
            }
        avg_test_acc = metrics.accuracy(
                np.sum(eval_tool.total_tp), np.sum(eval_tool.total_fp), np.sum(eval_tool.total_fn), np.sum(eval_tool.total_tn)).item()

        if avg_test_acc > self.max_acc:
            self.max_acc = avg_test_acc
            self.logger.info(f"-- Saving best model with testing accuracy {self.max_acc:.3f} --")
            checkpoint_name = 'ckpt_best.pth'
            torch.save(checkpoint, os.path.join(self.checkpoint_path, checkpoint_name))

        if self.epoch%self.config.train.checkpoint_saving_steps == 0:
            self.logger.info(f"Saving model with testing accuracy {avg_test_acc:.3f} in epoch {self.epoch} ")
            checkpoint_name = 'ckpt_best_{:04d}.pth'.format(self.epoch)
            torch.save(checkpoint, os.path.join(self.checkpoint_path, checkpoint_name))



if __name__ == '__main__':
    trainer = Trainer(model)