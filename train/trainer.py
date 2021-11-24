import os
import torch
from tensorboardX import SummaryWriter
# TODO: Try to Minimum libraries dependency


class Trainer(object):

    def __init__(self, 
                 model, 
                 criterion, 
                 optimizer, 
                 train_dataloader, 
                 valid_dataloader,
                 logger,
                 device,
                 USE_CUDA=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.logger = logger
        self.iterations = 0
        self.device = device
        self.USE_CUDA = USE_CUDA
        self.using_tensorboard = True
        if self.using_tensorboard:
            self.train_writer = SummaryWriter(log_dir=os.path.join(checkpoint_path, 'train'))
            self.test_writer = SummaryWriter(log_dir=os.path.join(checkpoint_path, 'valid'))
        
    def run(self, epochs=1):
        for self.cur_epoch in range(1, epochs + 1):
            self.train()
            with torch.no_grad():
                self.validate()
                self.save_model()
            
    def train(self):
        self.model.train()
        for i, data in enumerate(self.train_dataloader, self.iterations + 1):
            total_train_loss = 0.0
            # batch_input, batch_target = data
            # input_var = batch_input
            # target_var = batch_target
            input_var, target_var = data['input'], data['gt']
            input_var = input_var.to(self.device)
            target_var = target_var.to(self.device)
            def closure():
                batch_output = self.model(input_var)
                loss = self.criterion(batch_output, target_var)
                loss.backward()
                return loss
            self.optimizer.zero_grad()
            self.optimizer.step(closure)
        
            loss = loss.item()
            total_train_loss += loss
            if self.using_tensorboard:
                self.train_writer.add_scalar('Loss/step', loss, n_iter)

        self.iterations += i
        self.train_writer.add_scalar('Loss/epoch', total_train_loss/training_steps, self.cur_epoch)

    def validate(self):
        self.model.eval()
        eval_tool = metrics.SegmentationMetrics(config.model.out_channels, ['accuracy'])
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
            # test_writer.add_scalar('Loss/step', loss, test_n_iter)

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
            "epoch": self.cur_epoch
            }
            
        if avg_test_acc > max_acc:
            max_acc = avg_test_acc
            self.logger.info(f"-- Saving best model with testing accuracy {max_acc:.3f} --")
            checkpoint_name = 'ckpt_best.pth'
            torch.save(checkpoint, os.path.join(checkpoint_path, checkpoint_name))

        if self.cur_epoch%config.train.checkpoint_saving_steps == 0:
            self.logger.info(f"Saving model with testing accuracy {avg_test_acc:.3f} in epoch {self.cur_epoch} ")
            checkpoint_name = 'ckpt_best_{:04d}.pth'.format(self.cur_epoch)
            torch.save(checkpoint, os.path.join(checkpoint_path, checkpoint_name))



if __name__ == '__main__':
    trainer = Trainer(model)