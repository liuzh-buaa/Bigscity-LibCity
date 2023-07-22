from logging import getLogger

import numpy as np
import torch
import torch.nn as nn

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model.covert_dcrnn_to_b import convert_dcrnn_to_bdcrnn
from libcity.model.traffic_speed_prediction.layers.functions import count_parameters
from libcity.model.traffic_speed_prediction.layers.rand_dcgru_cell import RandDCGRUCell
from libcity.model.traffic_speed_prediction.layers.rand_linear import RandLinear


class Seq2SeqAttrs:
    def __init__(self, config, adj_mx):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(config.get('max_diffusion_step', 2))
        self.cl_decay_steps = int(config.get('cl_decay_steps', 1000))
        self.filter_type = config.get('filter_type', 'laplacian')
        self.num_nodes = int(config.get('num_nodes', 1))
        self.num_rnn_layers = int(config.get('num_rnn_layers', 2))
        self.rnn_units = int(config.get('rnn_units', 64))
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.input_dim = config.get('feature_dim', 1)
        self.device = config.get('device', torch.device('cpu'))
        self.sigma_pi = float(config.get('sigma_pi'))
        self.sigma_start = float(config.get('sigma_start'))
        self.sigma_0 = float(config.get('sigma_0'))
        self.reg_encoder = config.get('reg_encoder')
        self.reg_decoder = config.get('reg_decoder')
        self.loss_function = config.get('loss_function')


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, adj_mx)
        self.dcgru_layers = nn.ModuleList()
        self.dcgru_layers.append(RandDCGRUCell(self.input_dim, self.rnn_units, adj_mx, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type,
                                               sigma_pi=self.sigma_pi, sigma_start=self.sigma_start))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(RandDCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step,
                                                   self.num_nodes, self.device, filter_type=self.filter_type,
                                                   sigma_pi=self.sigma_pi, sigma_start=self.sigma_start))

    def forward(self, inputs, hidden_state=None):
        """
        Encoder forward pass.

        Args:
            inputs: shape (batch_size, self.num_nodes * self.input_dim)
            hidden_state: (num_layers, batch_size, self.hidden_state_size),
                optional, zeros if not provided, hidden_state_size = num_nodes * rnn_units

        Returns:
            tuple: tuple contains:
                output: shape (batch_size, self.hidden_state_size) \n
                hidden_state: shape (num_layers, batch_size, self.hidden_state_size) \n
                (lower indices mean lower layers)

        """
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size), device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            # next_hidden_state: (batch_size, self.num_nodes * self.rnn_units)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state  # 循环
        return output, torch.stack(hidden_states)  # runs in O(num_layers) so not too slow

    def get_kl_sum(self):
        kl_sum = 0
        for dcgru_layer in self.dcgru_layers:
            kl_sum += dcgru_layer.get_kl_sum()
        return kl_sum


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, adj_mx)
        self.output_dim = config.get('output_dim', 1)
        self.projection_layer = RandLinear(self.rnn_units, self.output_dim, self.device,
                                           sigma_pi=self.sigma_pi, sigma_start=self.sigma_start)
        self.dcgru_layers = nn.ModuleList()
        self.dcgru_layers.append(RandDCGRUCell(self.output_dim, self.rnn_units, adj_mx, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type,
                                               sigma_pi=self.sigma_pi, sigma_start=self.sigma_start))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(RandDCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step,
                                                   self.num_nodes, self.device, filter_type=self.filter_type,
                                                   sigma_pi=self.sigma_pi, sigma_start=self.sigma_start))

    def forward(self, inputs, hidden_state=None):
        """
        Decoder forward pass.

        Args:
            inputs:  shape (batch_size, self.num_nodes * self.output_dim)
            hidden_state: (num_layers, batch_size, self.hidden_state_size),
                optional, zeros if not provided, hidden_state_size = num_nodes * rnn_units

        Returns:
            tuple: tuple contains:
                output: shape (batch_size, self.num_nodes * self.output_dim) \n
                hidden_state: shape (num_layers, batch_size, self.hidden_state_size) \n
                (lower indices mean lower layers)
        """
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            # next_hidden_state: (batch_size, self.num_nodes * self.rnn_units)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)
        return output, torch.stack(hidden_states)

    def get_kl_sum(self):
        kl_sum = self.projection_layer.get_kl_sum()
        for dcgru_layer in self.dcgru_layers:
            kl_sum += dcgru_layer.get_kl_sum()
        return kl_sum


class BDCRNNConstant(AbstractTrafficStateModel, Seq2SeqAttrs):
    def __init__(self, config, data_feature):
        self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        config['num_nodes'] = self.num_nodes
        config['feature_dim'] = self.feature_dim
        self.output_dim = data_feature.get('output_dim', 1)

        super().__init__(config, data_feature)
        Seq2SeqAttrs.__init__(self, config, self.adj_mx)
        self.encoder_model = EncoderModel(config, self.adj_mx)
        self.decoder_model = DecoderModel(config, self.adj_mx)

        self.use_curriculum_learning = config.get('use_curriculum_learning', False)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        if config['init_params_from_dcrnn']:
            convert_dcrnn_to_bdcrnn(self, self.device)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps

        Args:
            inputs: shape (input_window, batch_size, num_sensor * input_dim)

        Returns:
            torch.tensor: (num_layers, batch_size, self.hidden_state_size)
        """
        encoder_hidden_state = None
        for t in range(self.input_window):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)
            # encoder_hidden_state: encoder的多层GRU的全部的隐层 (num_layers, batch_size, self.hidden_state_size)

        return encoder_hidden_state  # 最后一个隐状态

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        """
        Decoder forward pass

        Args:
            encoder_hidden_state: (num_layers, batch_size, self.hidden_state_size)
            labels:  (self.output_window, batch_size, self.num_nodes * self.output_dim)
                [optional, not exist for inference]
            batches_seen: global step [optional, not exist for inference]

        Returns:
            torch.tensor: (self.output_window, batch_size, self.num_nodes * self.output_dim)
        """
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.output_dim), device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []
        for t in range(self.output_window):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input, decoder_hidden_state)
            decoder_input = decoder_output  # (batch_size, self.num_nodes * self.output_dim)
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]  # (batch_size, self.num_nodes * self.output_dim)
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, batch, batches_seen=None):
        """
        seq2seq forward pass

        Args:
            batch: a batch of input,
                batch['X']: shape (batch_size, input_window, num_nodes, input_dim) \n
                batch['y']: shape (batch_size, output_window, num_nodes, output_dim) \n
            batches_seen: batches seen till now

        Returns:
            torch.tensor: (batch_size, self.output_window, self.num_nodes, self.output_dim)
        """
        inputs = batch['X']
        labels = batch['y']
        batch_size, _, num_nodes, input_dim = inputs.shape
        inputs = inputs.permute(1, 0, 2, 3)  # (input_window, batch_size, num_nodes, input_dim)
        inputs = inputs.view(self.input_window, batch_size, num_nodes * input_dim).to(self.device)
        self._logger.debug("X: {}".format(inputs.size()))  # (input_window, batch_size, num_nodes * input_dim)

        if labels is not None:
            labels = labels.permute(1, 0, 2, 3)  # (output_window, batch_size, num_nodes, output_dim)
            labels = labels[..., :self.output_dim].contiguous().view(
                self.output_window, batch_size, num_nodes * self.output_dim).to(self.device)
            self._logger.debug("y: {}".format(labels.size()))

        encoder_hidden_state = self.encoder(inputs)
        # (num_layers, batch_size, self.hidden_state_size)
        self._logger.debug("Encoder complete")
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        # (self.output_window, batch_size, self.num_nodes * self.output_dim)
        self._logger.debug("Decoder complete")

        if batches_seen == 0:
            self._logger.info("Total trainable parameters {}".format(count_parameters(self)))
        outputs = outputs.view(self.output_window, batch_size, self.num_nodes, self.output_dim).permute(1, 0, 2, 3)
        return outputs

    def calculate_loss(self, batch, batches_seen=None, num_batches=1):
        y_true = batch['y']
        y_predicted = self.predict(batch, batches_seen)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        if self.loss_function == 'masked_mae':
            return loss.masked_mae_const_reg_torch(y_predicted, y_true, self.sigma_0, self._get_kl_sum() / num_batches)
        elif self.loss_function == 'masked_mse':
            return loss.masked_mse_const_reg_torch(y_predicted, y_true, self.sigma_0, self._get_kl_sum() / num_batches)
        else:
            raise NotImplementedError('Unrecognized loss function.')

    def calculate_eval_loss(self, batch, batches_seen=None):
        y_true = batch['y']
        y_predicted = self.predict(batch, batches_seen)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch, batches_seen=None):
        return self.forward(batch, batches_seen)

    def predict_sigma(self, batch, batches_seen=None):
        return self.sigma_0

    def _get_kl_sum(self):
        kl_sum = 0
        if self.reg_encoder:
            kl_sum += self.encoder_model.get_kl_sum()
        if self.reg_decoder:
            kl_sum += self.decoder_model.get_kl_sum()
        return kl_sum
