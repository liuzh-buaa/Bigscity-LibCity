import math
from logging import getLogger

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import linalg

from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model.covert_dcrnn_to_b import convert_dcrnn_to_bdcrnn


def calculate_normalized_laplacian(adj):
    """
    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2

    Args:
        adj: adj matrix

    Returns:
        np.ndarray: L
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RandGCONV(nn.Module):
    def __init__(self, num_nodes, max_diffusion_step, supports, device, input_dim, hid_dim, output_dim, bias_start=0.0,
                 sigma_pi=1.0, sigma_start=1.0, init_func=torch.nn.init.xavier_normal_):
        super(RandGCONV, self).__init__()
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._supports = supports
        self._device = device
        self._num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Ks
        self._output_dim = output_dim
        self._sigma_pi = sigma_pi
        input_size = input_dim + hid_dim
        shape = (input_size * self._num_matrices, self._output_dim)
        self.mu_weight = torch.nn.Parameter(torch.empty(*shape, device=self._device))
        self.mu_biases = torch.nn.Parameter(torch.empty(self._output_dim, device=self._device))
        self.log_sigma_weight = torch.nn.Parameter(torch.empty(*shape, device=self._device))
        self.log_sigma_biases = torch.nn.Parameter(torch.empty(self._output_dim, device=self._device))
        # self.register_buffer('buffer_eps_weight', torch.empty(*shape, device=self._device))
        # self.register_buffer('buffer_eps_bias', torch.empty(self._output_dim, device=self._device))
        init_func(self.mu_weight)
        torch.nn.init.constant_(self.mu_biases, bias_start)
        torch.nn.init.constant_(self.log_sigma_weight, math.log(sigma_start))
        torch.nn.init.constant_(self.log_sigma_biases, math.log(sigma_start))
        # torch.nn.init.constant_(self.buffer_eps_weight, 0)
        # torch.nn.init.constant_(self.buffer_eps_bias, 0)

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, inputs, state):
        # 对X(t)和H(t-1)做图卷积，并加偏置bias
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        # (batch_size, num_nodes, total_arg_size(input_dim+state_dim))
        input_size = inputs_and_state.size(2)  # =total_arg_size

        x = inputs_and_state
        # T0=I x0=T0*x=x
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)  # (1, num_nodes, total_arg_size * batch_size)

        # 3阶[T0,T1,T2]Chebyshev多项式近似g(theta)
        # 把图卷积公式中的~L替换成了随机游走拉普拉斯D^(-1)*W
        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                # T1=L x1=T1*x=L*x
                x1 = torch.sparse.mm(support, x0)  # supports: n*n; x0: n*(total_arg_size * batch_size)
                x = self._concat(x, x1)  # (2, num_nodes, total_arg_size * batch_size)
                for k in range(2, self._max_diffusion_step + 1):
                    # T2=2LT1-T0=2L^2-1 x2=T2*x=2L^2x-x=2L*x1-x0...
                    # T3=2LT2-T1=2L(2L^2-1)-L x3=2L*x2-x1...
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)  # (3, num_nodes, total_arg_size * batch_size)
                    x1, x0 = x2, x1  # 循环
        # x.shape (Ks, num_nodes, total_arg_size * batch_size)
        # Ks = len(supports) * self._max_diffusion_step + 1

        x = torch.reshape(x, shape=[self._num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, num_matrices)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * self._num_matrices])

        sigma_weight = torch.exp(self.log_sigma_weight)
        weight = self.mu_weight + sigma_weight * torch.randn(self.mu_weight.shape, device=self._device)
        x = torch.matmul(x, weight)  # (batch_size * self._num_nodes, self._output_dim)
        sigma_bias = torch.exp(self.log_sigma_biases)
        bias = self.mu_biases + sigma_bias * torch.randn(self.mu_biases.shape, device=self._device)
        x = x + bias
        # Reshape res back to 2D: (batch_size * num_nodes, state_dim) -> (batch_size, num_nodes * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * self._output_dim])

    def get_kl_sum(self):
        kl_weight = math.log(self._sigma_pi) - self.log_sigma_weight + 0.5 * (
                torch.exp(self.log_sigma_weight) ** 2 + self.mu_weight ** 2) / (self._sigma_pi ** 2)
        kl_bias = math.log(self._sigma_pi) - self.log_sigma_biases + 0.5 * (
                torch.exp(self.log_sigma_biases) ** 2 + self.mu_biases ** 2) / (self._sigma_pi ** 2)
        return kl_weight.sum() + kl_bias.sum()


class RandFC(nn.Module):
    def __init__(self, num_nodes, device, input_dim, hid_dim, output_dim, bias_start=0.0,
                 sigma_pi=1.0, sigma_start=1.0, init_func=torch.nn.init.xavier_normal_):
        super(RandFC, self).__init__()
        self._num_nodes = num_nodes
        self._device = device
        self._output_dim = output_dim
        self._sigma_pi = sigma_pi
        input_size = input_dim + hid_dim
        shape = (input_size, self._output_dim)
        self.mu_weight = torch.nn.Parameter(torch.empty(*shape, device=self._device))
        self.mu_biases = torch.nn.Parameter(torch.empty(self._output_dim, device=self._device))
        self.log_sigma_weight = torch.nn.Parameter(torch.empty(*shape, device=self._device))
        self.log_sigma_biases = torch.nn.Parameter(torch.empty(self._output_dim, device=self._device))
        # self.register_buffer('buffer_eps_weight', torch.empty(*shape, device=self._device))
        # self.register_buffer('buffer_eps_bias', torch.empty(self._output_dim, device=self._device))
        init_func(self.mu_weight)
        torch.nn.init.constant_(self.mu_biases, bias_start)
        torch.nn.init.constant_(self.log_sigma_weight, math.log(sigma_start))
        torch.nn.init.constant_(self.log_sigma_biases, math.log(sigma_start))
        # torch.nn.init.constant_(self.buffer_eps_weight, 0)
        # torch.nn.init.constant_(self.buffer_eps_bias, 0)

    def forward(self, inputs, state):
        batch_size = inputs.shape[0]
        # Reshape input and state to (batch_size * self._num_nodes, input_dim/state_dim)
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        # (batch_size * self._num_nodes, input_size(input_dim+state_dim))
        sigma_weight = torch.exp(self.log_sigma_weight)
        weight = self.mu_weight + sigma_weight * torch.randn(self.mu_weight.shape, device=self._device)
        value = torch.sigmoid(torch.matmul(inputs_and_state, weight))
        # (batch_size * self._num_nodes, self._output_dim)
        sigma_bias = torch.exp(self.log_sigma_biases)
        bias = self.mu_biases + sigma_bias * torch.randn(self.mu_biases.shape, device=self._device)
        value = value + bias
        # Reshape res back to 2D: (batch_size * num_nodes, state_dim) -> (batch_size, num_nodes * state_dim)
        return torch.reshape(value, [batch_size, self._num_nodes * self._output_dim])

    def get_kl_sum(self):
        kl_weight = math.log(self._sigma_pi) - self.log_sigma_weight + 0.5 * (
                torch.exp(self.log_sigma_weight) ** 2 + self.mu_weight ** 2) / (self._sigma_pi ** 2)
        kl_bias = math.log(self._sigma_pi) - self.log_sigma_biases + 0.5 * (
                torch.exp(self.log_sigma_biases) ** 2 + self.mu_biases ** 2) / (self._sigma_pi ** 2)
        return kl_weight.sum() + kl_bias.sum()


class RandLinear(nn.Module):
    def __init__(self, in_features, out_features, device, bias=True, sigma_pi=1.0, sigma_start=1.0):
        super(RandLinear, self).__init__()

        self._sigma_pi = sigma_pi
        self._device = device

        self.in_features = in_features
        self.out_features = out_features
        self.mu_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        # self.register_buffer('buffer_eps_weight', torch.Tensor(out_features, in_features))
        if bias:
            self.mu_bias = torch.nn.Parameter(torch.Tensor(out_features))
            self.log_sigma_bias = torch.nn.Parameter(torch.Tensor(out_features))
            # self.register_buffer('buffer_eps_bias', torch.Tensor(out_features))
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('log_sigma_bias', None)
            # self.register_buffer('buffer_eps_bias', None)

        torch.nn.init.kaiming_uniform_(self.mu_weight, a=math.sqrt(5))
        torch.nn.init.constant_(self.log_sigma_weight, math.log(sigma_start))
        # torch.nn.init.constant_(self.buffer_eps_weight, 0)
        if self.mu_bias is not None:
            def _calculate_fan_in_and_fan_out(tensor):
                dimensions = tensor.dim()
                if dimensions < 2:
                    raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

                num_input_fmaps = tensor.size(1)
                num_output_fmaps = tensor.size(0)
                receptive_field_size = 1
                if tensor.dim() > 2:
                    receptive_field_size = tensor[0][0].numel()
                fan_in = num_input_fmaps * receptive_field_size
                fan_out = num_output_fmaps * receptive_field_size

                return fan_in, fan_out

            fan_in, _ = _calculate_fan_in_and_fan_out(self.mu_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.mu_bias, -bound, bound)
            torch.nn.init.constant_(self.log_sigma_bias, math.log(sigma_start))
            # torch.nn.init.constant_(self.buffer_eps_bias, 0)

    def forward(self, input):
        sigma_weight = torch.exp(self.log_sigma_weight)
        weight = self.mu_weight + sigma_weight * torch.randn(self.mu_weight.shape, device=self._device)
        if self.mu_bias is not None:
            sigma_bias = torch.exp(self.log_sigma_bias)
            bias = self.mu_bias + sigma_bias * torch.randn(self.mu_bias.shape, device=self._device)
        else:
            bias = None
        return F.linear(input, weight, bias)

    def get_kl_sum(self):
        kl_weight = math.log(self._sigma_pi) - self.log_sigma_weight + 0.5 * (
                torch.exp(self.log_sigma_weight) ** 2 + self.mu_weight ** 2) / (self._sigma_pi ** 2)
        if self.mu_bias is not None:
            kl_bias = math.log(self._sigma_pi) - self.log_sigma_bias + 0.5 * (
                    torch.exp(self.log_sigma_bias) ** 2 + self.mu_bias ** 2) / (self._sigma_pi ** 2)
        else:
            kl_bias = 0

        return kl_weight.sum() + kl_bias.sum()


class RandDCGRUCell(nn.Module):
    def __init__(self, input_dim, num_units, adj_mx, max_diffusion_step, num_nodes, device, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True, sigma_pi=1.0, sigma_start=1.0,
                 init_func=torch.nn.init.xavier_normal_):
        """

        Args:
            input_dim:
            num_units:
            adj_mx:
            max_diffusion_step:
            num_nodes:
            device:
            nonlinearity:
            filter_type: "laplacian", "random_walk", "dual_random_walk"
            use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """

        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._device = device
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru

        supports = []
        if filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
            supports.append(calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support, self._device))

        if self._use_gc_for_ru:
            self._fn = RandGCONV(self._num_nodes, self._max_diffusion_step, self._supports, self._device,
                                 input_dim=input_dim, hid_dim=self._num_units, output_dim=2 * self._num_units,
                                 bias_start=1.0, sigma_pi=sigma_pi, sigma_start=sigma_start, init_func=init_func)
        else:
            self._fn = RandFC(self._num_nodes, self._device, input_dim=input_dim,
                              hid_dim=self._num_units, output_dim=2 * self._num_units, bias_start=1.0,
                              sigma_pi=sigma_pi, sigma_start=sigma_start, init_func=init_func)
        self._gconv = RandGCONV(self._num_nodes, self._max_diffusion_step, self._supports, self._device,
                                input_dim=input_dim, hid_dim=self._num_units, output_dim=self._num_units,
                                bias_start=0.0,
                                sigma_pi=sigma_pi, sigma_start=sigma_start, init_func=init_func)

    @staticmethod
    def _build_sparse_matrix(lap, device):
        lap = lap.tocoo()
        indices = np.column_stack((lap.row, lap.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        lap = torch.sparse_coo_tensor(indices.T, lap.data, lap.shape, device=device)
        return lap

    def forward(self, inputs, hx):
        """
        Gated recurrent unit (GRU) with Graph Convolution.

        Args:
            inputs: (B, num_nodes * input_dim)
            hx: (B, num_nodes * rnn_units)

        Returns:
            torch.tensor: shape (B, num_nodes * rnn_units)
        """
        output_size = 2 * self._num_units
        value = torch.sigmoid(self._fn(inputs, hx))  # (batch_size, num_nodes * output_size)
        value = torch.reshape(value, (-1, self._num_nodes, output_size))  # (batch_size, num_nodes, output_size)

        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))  # (batch_size, num_nodes * _num_units)
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))  # (batch_size, num_nodes * _num_units)

        c = self._gconv(inputs, r * hx)  # (batch_size, num_nodes * _num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state  # (batch_size, num_nodes * _num_units)

    def get_kl_sum(self):
        return self._fn.get_kl_sum() + self._gconv.get_kl_sum()


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
        self.sigma_sigma_pi = float(config.get('sigma_sigma_pi'))
        self.sigma_sigma_start = float(config.get('sigma_sigma_start'))
        self.reg_encoder = config.get('reg_encoder')
        self.reg_decoder = config.get('reg_decoder')
        self.reg_encoder_sigma_0 = config.get('reg_encoder_sigma_0')
        self.reg_decoder_sigma_0 = config.get('reg_decoder_sigma_0')
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


class EncoderSigmaModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, adj_mx)
        self.dcgru_layers = nn.ModuleList()

        def init_func_xavier_normal_1_10(tensor):
            return torch.nn.init.xavier_normal_(tensor, gain=0.1)

        self.dcgru_layers.append(RandDCGRUCell(self.input_dim, self.rnn_units, adj_mx, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type,
                                               sigma_pi=self.sigma_sigma_pi, sigma_start=self.sigma_sigma_start))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(RandDCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step,
                                                   self.num_nodes, self.device, filter_type=self.filter_type,
                                                   sigma_pi=self.sigma_sigma_pi, sigma_start=self.sigma_sigma_start))

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


class DecoderSigmaModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, config, adj_mx):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, config, adj_mx)
        self.output_dim = config.get('output_dim', 1)
        self.projection_layer = RandLinear(self.rnn_units, self.output_dim, self.device,
                                           sigma_pi=self.sigma_sigma_pi, sigma_start=self.sigma_sigma_start)
        self.dcgru_layers = nn.ModuleList()

        def init_func_xavier_normal_1_10(tensor):
            return torch.nn.init.xavier_normal_(tensor, gain=0.1)

        self.dcgru_layers.append(RandDCGRUCell(self.output_dim, self.rnn_units, adj_mx, self.max_diffusion_step,
                                               self.num_nodes, self.device, filter_type=self.filter_type,
                                               sigma_pi=self.sigma_sigma_pi, sigma_start=self.sigma_sigma_start))
        for i in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(RandDCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step,
                                                   self.num_nodes, self.device, filter_type=self.filter_type,
                                                   sigma_pi=self.sigma_sigma_pi, sigma_start=self.sigma_sigma_start))

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


class BDCRNNLogVariable(AbstractTrafficStateModel, Seq2SeqAttrs):
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

        self.encoder_sigma_model = EncoderSigmaModel(config, self.adj_mx)
        self.decoder_sigma_model = DecoderSigmaModel(config, self.adj_mx)

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

    def encoder_sigma(self, inputs):
        encoder_hidden_state = None
        for t in range(self.input_window):
            _, encoder_hidden_state = self.encoder_sigma_model(inputs[t], encoder_hidden_state)
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

    def decoder_sigma(self, encoder_hidden_state, labels=None, batches_seen=None):
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.output_dim), device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []
        for t in range(self.output_window):
            decoder_output, decoder_hidden_state = self.decoder_sigma_model(decoder_input, decoder_hidden_state)
            decoder_input = decoder_output  # (batch_size, self.num_nodes * self.output_dim)
            outputs.append(decoder_output)
            # if self.training and self.use_curriculum_learning:
            #     c = np.random.uniform(0, 1)
            #     if c < self._compute_sampling_threshold(batches_seen):
            #         decoder_input = labels[t]  # (batch_size, self.num_nodes * self.output_dim)
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

    def forward_sigma(self, batch, batches_seen=None):
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

        encoder_hidden_state = self.encoder_sigma(inputs)
        # (num_layers, batch_size, self.hidden_state_size)
        self._logger.debug("Encoder sigma complete")
        outputs = self.decoder_sigma(encoder_hidden_state, labels, batches_seen=batches_seen)
        # (self.output_window, batch_size, self.num_nodes * self.output_dim)
        self._logger.debug("Decoder sigma complete")

        if batches_seen == 0:
            self._logger.info("Total trainable parameters {}".format(count_parameters(self)))
        outputs = outputs.view(self.output_window, batch_size, self.num_nodes, self.output_dim).permute(1, 0, 2, 3)
        return outputs

    def calculate_loss(self, batch, batches_seen=None):
        y_true = batch['y']
        y_predicted = self.predict(batch, batches_seen)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        log_sigma_0 = self.forward_sigma(batch, batches_seen)
        if self.loss_function == 'masked_mae':
            return loss.masked_mae_log_reg_torch(y_predicted, y_true, log_sigma_0, self._get_kl_sum(), 0)
        elif self.loss_function == 'masked_mse':
            return loss.masked_mse_log_reg_torch(y_predicted, y_true, log_sigma_0, self._get_kl_sum(), 0)
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

    def _get_kl_sum(self):
        kl_sum = 0
        if self.reg_encoder:
            kl_sum += self.encoder_model.get_kl_sum()
        if self.reg_decoder:
            kl_sum += self.decoder_model.get_kl_sum()
        if self.reg_encoder_sigma_0:
            kl_sum += self.encoder_sigma_model.get_kl_sum()
        if self.reg_decoder_sigma_0:
            kl_sum += self.decoder_sigma_model.get_kl_sum()
        return kl_sum


if __name__ == '__main__':
    RandGCONV(207, 2, [torch.randn(207, 207, device='cuda:0'), torch.randn(207, 207, device='cuda:0')], 'cuda:0',
              input_dim=2, hid_dim=64, output_dim=128, bias_start=1.0)
    RandFC(207, 'cuda:0', input_dim=2, hid_dim=64, output_dim=128, bias_start=1.0)
    print(torch.mean(torch.stack([torch.randn(3, 4, 5) for i in range(5)]), dim=0).shape)
