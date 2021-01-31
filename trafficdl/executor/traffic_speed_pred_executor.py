import os
import time
import numpy as np
import torch
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter
from trafficdl.executor.abstract_executor import AbstractExecutor
from trafficdl.utils import get_evaluator, ensure_dir


class TrafficSpeedPredExecutor(AbstractExecutor):
    def __init__(self, config, model):
        self.evaluator = get_evaluator(config)
        self.config = config
        self.model = model.to(self.config['device'])
        self.metrics = self.config.get('metrics', 'MAE')

        self.tmp_path = './trafficdl/tmp/checkpoint'
        self.cache_dir = './trafficdl/cache/model_cache'
        self.evaluate_res_dir = './trafficdl/cache/evaluate_cache'
        self.summary_writer_dir = './trafficdl/log/runs'
        ensure_dir(self.tmp_path)
        ensure_dir(self.cache_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)

        self._writer = SummaryWriter(self.summary_writer_dir)
        self._logger = getLogger()
        self._scaler = self.model.get_data_feature().get('scaler')

        self.epochs = self.config.get('epochs', 100)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.weight_decay = self.config.get('weight_decay', 0)
        self.learner = self.config.get('learner', 'adam')
        self.epsilon = self.config.get('epsilon', 1e-8)
        self.lr_scheduler_type = self.config.get('lr_scheduler', 'multisteplr')
        self.lr_decay = self.config.get('lr_decay', False)
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.1)
        self.milestones = self.config.get('steps', [])
        self.step_size = self.config.get('step_size', 10)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.)
        self.clip_grad_norm = self.config.get('clip_grad_norm', False)
        self.log_every = self.config.get('log_every', 1)
        self.saved = self.config.get('save_model', True)
        self.patience = self.config.get('patience', 50)
        self.device = self.config.get('device', torch.device('cpu'))
        self.output_dim = config.get('output_dim', 1)
        self._epoch_num = self.config.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model_with_epoch(self._epoch_num)

        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

    def save_model(self, cache_name):
        ensure_dir(self.cache_dir)
        self._logger.info("Saved model at " + cache_name)
        torch.save(self.model.state_dict(), cache_name)

    def load_model(self, cache_name):
        self._logger.info("Loaded model at " + cache_name)
        self.model.load_state_dict(torch.load(cache_name))

    def save_model_with_epoch(self, epoch):
        ensure_dir(self.cache_dir)
        config = dict(self.config)
        config['model_state_dict'] = self.model.state_dict()
        config['epoch'] = epoch
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        torch.save(config, model_path)
        self._logger.info("Saved model at {}".format(epoch))
        return model_path

    def load_model_with_epoch(self, epoch):
        model_path = self.cache_dir + '/' + self.config['model'] + '_' + self.config['dataset'] + '_epoch%d.tar' % epoch
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch))

    def _build_optimizer(self):
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.epsilon, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.learning_rate)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=self.epsilon)
        return optimizer

    def _build_lr_scheduler(self):
        if self.lr_decay:
            if self.lr_scheduler_type.lower() == 'multisteplr':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.milestones, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'steplr':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.step_size, gamma=self.lr_decay_ratio)
            else:
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler

    def evaluate(self, test_dataloader):
        self._logger.info('Start evaluating ...')
        with torch.no_grad():
            self.model.eval()
            self.evaluator.clear()
            y_truths = []
            y_preds = []
            for batch in test_dataloader:
                batch.to_tensor(self.device)
                output = self.model.predict(batch)
                y_true = self._scaler.inverse_transform(batch['y'][..., :self.output_dim])
                y_pred = self._scaler.inverse_transform(output[..., :self.output_dim])
                y_truths.append(y_true.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())
                evaluate_input = {'y_true': y_true, 'y_pred': y_pred}
                self.evaluator.collect(evaluate_input)
            self.evaluator.save_result(self.evaluate_res_dir)
            y_preds = np.concatenate(y_preds, axis=0)
            y_truths = np.concatenate(y_truths, axis=0)  # concatenate on batch
            outputs = {'prediction': y_preds, 'truth': y_truths}
            filename = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) \
                       + '_' + self.config['model'] + '_predictions.npz'
            np.savez_compressed(os.path.join(self.evaluate_res_dir, filename), **outputs)
            self.evaluator.clear()
            self.evaluator.collect({'y_true': torch.tensor(y_truths), 'y_pred': torch.tensor(y_preds)})
            self.evaluator.save_result(self.evaluate_res_dir)

    def train(self, train_dataloader, eval_dataloader):
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        num_batches = len(train_dataloader)
        self._logger.info("num_batches:{}".format(num_batches))

        for epoch_idx in range(self._epoch_num, self.epochs):
            start_time = time.time()
            losses = self._train_epoch(train_dataloader, epoch_idx)
            self._writer.add_scalar('training loss', np.mean(losses), epoch_idx)
            self._logger.info("epoch complete!")
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self._logger.info("evaluating now!")
            val_loss = self._valid_epoch(eval_dataloader, epoch_idx)
            end_time = time.time()

            if (epoch_idx % self.log_every) == 0:
                if self.lr_scheduler is not None:
                    log_lr = self.lr_scheduler.get_last_lr()[0]
                else:
                    log_lr = self.learning_rate
                message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.1f}s'.\
                    format(epoch_idx, self.epochs, np.mean(losses), val_loss, log_lr, (end_time - start_time))
                self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break
        self.load_model_with_epoch(best_epoch)

    def _train_epoch(self, train_dataloader, epoch_idx, loss_func=None):
        self.model.train()
        loss_func = loss_func if loss_func is not None else self.model.calculate_loss
        losses = []
        for batch in train_dataloader:
            self.optimizer.zero_grad()
            batch.to_tensor(self.device)
            loss = loss_func(batch)
            self._logger.debug(loss.item())
            losses.append(loss.item())
            loss.backward()
            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        return losses

    def _valid_epoch(self, eval_dataloader, epoch_idx, loss_func=None):
        with torch.no_grad():
            self.model.eval()
            loss_func = loss_func if loss_func is not None else self.model.calculate_loss
            losses = []
            for batch in eval_dataloader:
                batch.to_tensor(self.device)
                loss = loss_func(batch)
                self._logger.debug(loss.item())
                losses.append(loss.item())
            mean_loss = np.mean(losses)
            self._writer.add_scalar('eval loss', mean_loss, epoch_idx)
            return mean_loss

