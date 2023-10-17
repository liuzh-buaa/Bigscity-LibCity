import math
import numbers
import warnings

import torch
from torch import nn, Tensor
from torch.nn import Parameter, init

# TODO: not finished
class RandRNNBase(nn.Module):
    __constants__ = ['mode', 'input_size', 'hidden_size', 'num_layers', 'bias',
                     'batch_first', 'dropout', 'bidirectional']
    __jit_unused_properties__ = ['all_weights']

    mode: str
    input_size: int
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout: float
    bidirectional: bool

    def __init__(self, mode: str, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, batch_first: bool = False,
                 dropout: float = 0., bidirectional: bool = False, sigma_pi=1.0, sigma_start=1.0):
        super(RandRNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self._sigma_pi = sigma_pi
        num_directions = 2 if bidirectional else 1

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        elif mode == 'RNN_TANH':
            gate_size = hidden_size
        elif mode == 'RNN_RELU':
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self._flat_weights_names = []
        self._all_weights = []
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions

                mu_w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                init.uniform_(mu_w_ih, -stdv, stdv)
                log_sigma_w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                init.constant_(log_sigma_w_ih, math.log(sigma_start))
                mu_w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                init.uniform_(mu_w_hh, -stdv, stdv)
                log_sigma_w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                init.constant_(log_sigma_w_hh, math.log(sigma_start))
                mu_b_ih = Parameter(torch.Tensor(gate_size))
                init.uniform_(mu_b_ih, -stdv, stdv)
                log_sigma_b_ih = Parameter(torch.Tensor(gate_size))
                init.constant_(log_sigma_b_ih, math.log(sigma_start))
                # Second bias vector included for CuDNN compatibility. Only one
                # bias vector is needed in standard definition.
                mu_b_hh = Parameter(torch.Tensor(gate_size))
                init.uniform_(mu_b_hh, -stdv, stdv)
                log_sigma_b_hh = Parameter(torch.Tensor(gate_size))
                init.constant_(log_sigma_b_hh, math.log(sigma_start))
                layer_params = (
                    mu_w_ih, log_sigma_w_ih, mu_w_hh, log_sigma_w_hh, mu_b_ih, log_sigma_b_ih, mu_b_hh, log_sigma_b_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['mu_weight_ih_l{}{}', 'log_sigma_weight_ih_l{}{}',
                               'mu_weight_hh_l{}{}', 'log_sigma_weight_hh_l{}{}']
                if bias:
                    param_names += ['mu_bias_ih_l{}{}', 'log_sigma_bias_ih_l{}{}',
                                    'mu_bias_hh_l{}{}', 'log_sigma_bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._flat_weights_names.extend(param_names)
                self._all_weights.append(param_names)

        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in
                              self._flat_weights_names]
        self.flatten_parameters()
        # self.reset_parameters()

        self.shared_eps = False

    def __setattr__(self, attr, value):
        if hasattr(self, "_flat_weights_names") and attr in self._flat_weights_names:
            # keep self._flat_weights up to date if you do self.weight = ...
            idx = self._flat_weights_names.index(attr)
            self._flat_weights[idx] = value
        super(RandRNNBase, self).__setattr__(attr, value)

    def flatten_parameters(self) -> None:
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        # Short-circuits if _flat_weights is only partially instantiated
        if len(self._flat_weights) != len(self._flat_weights_names):
            return

        for w in self._flat_weights:
            if not isinstance(w, Tensor):
                return
        # Short-circuits if any tensor in self._flat_weights is not acceptable to cuDNN
        # or the tensors in _flat_weights are of different dtypes

        first_fw = self._flat_weights[0]
        dtype = first_fw.dtype
        for fw in self._flat_weights:
            if (not isinstance(fw.data, Tensor) or not (fw.data.dtype == dtype) or
                    not fw.data.is_cuda or
                    not torch.backends.cudnn.is_acceptable(fw.data)):
                return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        unique_data_ptrs = set(p.data_ptr() for p in self._flat_weights)
        if len(unique_data_ptrs) != len(self._flat_weights):
            return

        with torch.cuda.device_of(first_fw):
            import torch.backends.cudnn.rnn as rnn

            # Note: no_grad() is necessary since _cudnn_rnn_flatten_weight is
            # an inplace operation on self._flat_weights
            with torch.no_grad():
                if torch._use_cudnn_rnn_flatten_weight():
                    torch._cudnn_rnn_flatten_weight(
                        self._flat_weights, (4 if self.bias else 2),
                        self.input_size, rnn.get_cudnn_mode(self.mode), self.hidden_size, self.num_layers,
                        self.batch_first, bool(self.bidirectional))


    def _apply(self, fn):
        ret = super(RandRNNBase, self)._apply(fn)

        # Resets _flat_weights
        # Note: be v. careful before removing this, as 3rd party device types
        # likely rely on this behavior to properly .to() modules like LSTM.
        self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
        # Flattens params (on CUDA)
        self.flatten_parameters()

        return ret
