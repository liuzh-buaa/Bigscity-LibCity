import numpy as np
import torch

from libcity.model import loss
from libcity.model.traffic_speed_prediction.BDCRNNBase import BDCRNNBase
from libcity.model.traffic_speed_prediction.layers.functions import count_parameters


class BDCRNNVariableSharedLayer(BDCRNNBase):
    def __init__(self, config, data_feature):
        super(BDCRNNVariableSharedLayer, self).__init__(config, data_feature)

    def encoder(self, inputs):
        """
        encoder forward pass on t time steps

        Args:
            inputs: shape (input_window, batch_size, num_sensor * input_dim)

        Returns:
            torch.tensor: (num_layers, batch_size, self.hidden_state_size)
        """
        self.encoder_model.set_shared_eps()
        encoder_hidden_state = None
        for t in range(self.input_window):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)
            # encoder_hidden_state: encoder的多层GRU的全部的隐层 (num_layers, batch_size, self.hidden_state_size)
        self.encoder_model.clear_shared_eps()

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

        self.decoder_model.set_shared_eps()

        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.output_dim), device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs, sigma_0 = [], []
        for t in range(self.output_window):
            decoder_output, decoder_hidden_state, variance = self.decoder_model(decoder_input, decoder_hidden_state)
            decoder_input = decoder_output  # (batch_size, self.num_nodes * self.output_dim)
            outputs.append(decoder_output)
            sigma_0.append(variance)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]  # (batch_size, self.num_nodes * self.output_dim)
        outputs = torch.stack(outputs)
        sigma_0 = torch.stack(sigma_0)

        self.decoder_model.clear_shared_eps()

        return outputs, sigma_0

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
        outputs, sigma_0 = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        # (self.output_window, batch_size, self.num_nodes * self.output_dim)
        self._logger.debug("Decoder complete")

        if batches_seen == 0:
            self._logger.info("Total trainable parameters {}".format(count_parameters(self)))
        outputs = outputs.view(self.output_window, batch_size, self.num_nodes, self.output_dim).permute(1, 0, 2, 3)
        sigma_0 = sigma_0.view(self.output_window, batch_size, self.num_nodes, 1).permute(1, 0, 2, 3)
        return outputs, sigma_0

    def calculate_loss(self, batch, batches_seen=None, num_batches=1):
        y_true = batch['y']
        y_predicted, sigma_0 = self.forward(batch, batches_seen)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        ll = self.clamp_function.split('_')
        if self.loss_function == 'masked_mae' and ll[0] == 'relu':
            return loss.masked_mae_relu_reg_torch(y_predicted, y_true, sigma_0, self._get_kl_sum() / num_batches, 0, float(ll[1]))
        elif self.loss_function == 'masked_mae' and ll[0] == 'Softplus':
            return loss.masked_mae_softplus_reg_torch(y_predicted, y_true, sigma_0, self._get_kl_sum() / num_batches, 0, int(ll[1]))
        elif self.loss_function == 'masked_mse' and ll[0] == 'relu':
            return loss.masked_mse_relu_reg_torch(y_predicted, y_true, sigma_0, self._get_kl_sum() / num_batches, 0, float(ll[1]))
        elif self.loss_function == 'masked_mse' and ll[0] == 'Softplus':
            return loss.masked_mse_softplus_reg_torch(y_predicted, y_true, sigma_0, self._get_kl_sum() / num_batches, 0, int(ll[1]))
        else:
            raise NotImplementedError('Unrecognized loss function.')

    def predict(self, batch, batches_seen=None):
        return self.forward(batch, batches_seen)[0]

    def predict_sigma(self, batch, batches_seen=None):
        ll = self.clamp_function.split('_')
        if ll[0] == 'relu':
            return torch.clamp(self.forward(batch, batches_seen)[1], min=float(ll[1]))
        elif ll[0] == 'Softplus':
            return torch.nn.Softplus(beta=int(ll[1]))(self.forward(batch, batches_seen)[1])
        else:
            raise NotImplementedError('Unrecognized loss function.')
