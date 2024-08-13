# Implementation from the official code repo of:
# "Data-driven modeling of interrelated dynamical systems", by
# Yonatan Elul, Eyal Rozenberg, Amit Boyarski, Yael Yaniv, Assaf Schuster & Alex M. Bronstein
#
# Official code implementation: https://github.com/YonatanElul/midst.git


from torch import Tensor, cat, bmm
from abc import abstractmethod, ABC
from lldm.models.patchtst import RevIN
from typing import Union, Dict, Sequence, Tuple
from lldm.utils.defaults import MODELS_TENSOR_PREDICITONS_KEY, OTHER_KEY

import torch
import torch.nn as nn


class Snake(nn.Module):
    """
    The Snake activation function from:
    "Neural Networks Fail to Learn Periodic Functions and How to Fix It" - Liu Ziyin, Tilman Hartwig, Masahito Ueda
    """

    def __init__(self, a: float = 1.):
        """

        :param a: (float) The frequency of the periodic basis function
        """

        super().__init__()

        self._a = a
        self._b = (1 / (2 * a))

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the non-linear Snake activation function to x.

        :param x: The input Tensor

        :return: Snake(x)
        """

        out = x if self._a == 0 else (x - (self._b * (2 * self._a * x).cos()) + self._b)
        return out

    def __call__(self, x: Tensor):
        return self.forward(x)


class Swish(nn.Module):
    def __init__(self, b: float = 1.):
        super().__init__()
        self._b = b

    def forward(self, x: Tensor) -> Tensor:
        y = x / (1 + (-self._b * x).exp())
        return y

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


def init_weights(module: nn.Module):
    """
    Utility method for initializing weights in layers

    :param module: (PyTorch Module) The layer to be initialized.

    :return: None
    """

    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)


def get_activation_layer(activation: Union[str, dict] = 'relu') -> nn.Module:
    """
    A utility method for defining and instantiating a PyTorch
    non-linear activation layer.

    :param activation: (str / dict / None) non-linear activation function to apply.
    If a string is given, using the layer with default parameters.
    if dict is given uses the 'name' key to determine which activation function to
    use and the 'params' key should have a dict with the required parameters as a
    key-value pairs. Currently supported activations: 'relu', 'gelu', 'elu',
    'hardshrink', 'leakyrelu', 'prelu', 'tanh', 'snake', 'nlsq', default = 'relu'

    :return: (PyTorch Module) The activation layer.
    """

    # Define the activations keys-modules pairs
    activations_dict = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'elu': nn.ELU,
        'hardshrink': nn.Hardshrink,
        'leakyrelu': nn.LeakyReLU,
        'prelu': nn.PReLU,
        'tanh': nn.Tanh,
        'snake': Snake,
        'swish': Swish,
        'sigmoid': nn.Sigmoid
    }

    # Validate inputs
    if (activation is not None and not isinstance(activation, str)
            and not isinstance(activation, dict)):
        raise ValueError("Can't take specs for the activation layer of type"
                         f" {type(activation)}, please specify either with a "
                         "string or a dictionary.")

    if (isinstance(activation, dict)
            and activation['name'] not in activations_dict.keys()):
        raise ValueError(f"{activation['name']} is not a supported Activation type.")

    if isinstance(activation, str):
        activation_block = activations_dict[activation]()

    else:
        activation_block = activations_dict[activation['name']](
            **activation['params'])

    return activation_block


def get_normalization_layer(norm: dict) -> nn.Module:
    """
    A utility method for defining and instantiating a PyTorch normalization layer.

    :param norm: (dict / None) Denotes the normalization layer to use with the FC layer.
    The dict should contain at least two keys, 'name' for indicating the type of
    normalization to use, and 'params', which should also map to a dict with all
    required parameters for the normalization layer. At the minimum, the 'params' dict
    should define the 'num_channels' key to indicate the expected number of
    channels on which to apply the normalization. For the GroupNorm, it is also
    required to specify a 'num_groups' key.
    If None then doesn't add normalization layer.
    Currently supported normalization layers: 'batch1d', 'batch2d', 'batch3d',
    'instance1d', 'instance2d', 'instance3d', 'group', where 'batch' stands for
    BatchNorm, `instance` stands for InstanceNorm and `group` stands
    for GroupNorm. Default == None.

    :return: (PyTorch Module) The normalization layer.
    """

    # Define the normalizations keys-modules pairs
    norms_dict = {
        'batch1d': nn.BatchNorm1d,
        'batch2d': nn.BatchNorm2d,
        'batch3d': nn.BatchNorm3d,
        'instance1d': nn.InstanceNorm1d,
        'instance2d': nn.InstanceNorm2d,
        'instance3d': nn.InstanceNorm3d,
        'layer': nn.LayerNorm,
        'group': nn.GroupNorm,
    }

    # Validate inputs
    if norm is not None and not isinstance(norm, dict):
        raise ValueError(f"Can't specify norm as a {type(norm)} type. Please "
                         f"either use a dict, or None.")

    if (isinstance(norm, dict)
            and norm['name'] not in norms_dict.keys()):
        raise ValueError(f"{norm['name']} is not a supported Normalization type.")

    norm_block = norms_dict[norm['name']](
        **norm['params'])

    return norm_block


def get_fc_layer(
        input_dim: int,
        output_dim: int,
        bias: bool = False,
        activation: Union[str, dict, None] = 'relu',
        dropout: Union[float, None] = None,
        norm: Union[dict, None] = None,
) -> nn.Module:
    """
    A utility method for generating a FC layer

    :param input_dim: (int) input dimension of the 2D matrices
    (M for a N X M matrix)
    :param output_dim: (int) output dimension of the 2D matrices
     (M for a N X M matrix)
    :param bias: (bool) whether to use a bias in the FC layer or not,
     default = False
    :param activation: (str / dict / None) non-linear activation function to apply.
    If a string is given, using the layer with default parameters.
    if dict is given uses the 'name' key to determine which activation function to
    use and the 'params' key should have a dict with the required parameters as a
    key-value pairs. Currently supported activations: 'relu', 'gelu', 'elu',
    'hardshrink', 'leakyrelu', 'prelu', 'tanh', default = 'relu'
    :param dropout: (float/ None) rate of dropout to apply to the FC layer,
    if None than doesn't apply dropout, default = None
    :param norm: (dict / None) Denotes the normalization layer to use with the FC layer.
    The dict should contain at least two keys, 'name' for indicating the type of
    normalization to use, and 'params', which should also map to a dict with all
    required parameters for the normalization layer. At the minimum, the 'params' dict
    should define the 'num_channels' key to indicate the expected number of
    channels on which to apply the normalization. For the GroupNorm, it is also
    required to specify a 'num_groups' key.
    If None then doesn't add normalization layer.
    Currently supported normalization layers: 'batch1d', 'batch2d', 'batch3d',
    'instance1d', 'instance2d', 'instance3d', 'group', where 'batch' stands for
    BatchNorm, `instance` stands for InstanceNorm and `group` stands
    for GroupNorm. Default == None.

    :return: (PyTorch Module) the instantiated layer, according to the given specs
    """

    # Validate inputs
    if dropout is not None and (not isinstance(dropout, float) and not isinstance(dropout, int)):
        raise ValueError(f"Can't specify dropout as a {type(dropout)} type. Please "
                         f"either use float, or None.")

    # Add the FC block
    blocks = []
    fc_block = nn.Linear(
        in_features=input_dim,
        out_features=output_dim,
        bias=bias,
    )
    blocks.append(fc_block)

    # Add the Normalization block if required
    if norm is not None:
        norm_block = get_normalization_layer(norm)
        blocks.append(norm_block)

    # Add the Activation block if required
    if activation is not None:
        activation_block = get_activation_layer(activation)
        blocks.append(activation_block)

    # Add the Dropout block if required
    if dropout is not None:
        dropout_block = nn.Dropout(p=dropout)
        blocks.append(dropout_block)

    # Encapsulate all blocks as a single `Sequential` module.
    fc_layer = nn.Sequential(*blocks)

    return fc_layer


class SkipConnectionFCBlock(nn.Module):
    """
    FC block which applies skip connection.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            bias: bool = False,
            activation: Union[str, dict, None] = 'relu',
            dropout: Union[float, None] = None,
            norm: Union[dict, None] = None,
    ):
        super(SkipConnectionFCBlock, self).__init__()

        self._fc_block = get_fc_layer(
            input_dim=input_dim,
            output_dim=output_dim,
            bias=bias,
            activation=activation,
            dropout=dropout,
            norm=norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self._fc_block(x)
        out = out + x
        return out

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class FCEncoderDecoder(nn.Module, ABC):
    """
    Fully connected encoder/decoder model.
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int = 64,
            n_layers: int = 8,
            l0_units: int = 1024,
            units_factor: float = 0.5,
            units_per_layer: tuple = None,
            activation: Union[str, dict] = 'relu',
            final_activation: Union[str, dict] = None,
            norm: Union[str, dict] = None,
            dropout: float = None,
            bias: bool = False,
            skip_connections: bool = False,
    ):
        """
        The constructor of the FCEncoderDecoder class, note that this class can
        save as both encoder and decoder FC-based models.

        :param input_dim: (int) Either the dimensionality of the input 2D matrix,
        i.e. if the inputs is a matrix of size m X n, then `input_dim` is n, when
        using the model as an encoder, or the dimensionality of the latent
        representation in the embedding space when using the model as a decoder.
        :param output_dim: (int) Required dimensionality for the latent embeddings when
        using the model as an encoder, or the dimensionality of the final output
        when using the model as a decoder.
        :param n_layers: (int) Number of FC layers to include in the encoder.
        :param l0_units: (int) Number of units to include in the first FC layer.
        :param units_factor: (float) Multiplicative factor for reducing/increasing
        the number of units in each consecutive FC layer when using the model as
        encoder / decoder.
        :param units_per_layer: (tuple) A tuple with 'n_layers' elements, where each
        element i is an integer indicating  the number of units to use in the i-th FC
        layer. Must specify either 'units_per_layer',
        or 'l0_units' and 'units_down_factor'.
        :param bias: (bool) whether to use a bias in the FC layer or not,
         default = False
        :param activation: (str / dict / None) non-linear activation function to apply.
        If a string is given, using the layer with default parameters.
        if dict is given uses the 'name' key to determine which activation function to
        use and the 'params' key should have a dict with the required parameters as a
        key-value pairs. Currently supported activations: 'relu', 'gelu', 'elu',
        'hardshrink', 'leakyrelu', 'prelu', 'tanh', default = 'relu'.
        :param final_activation: (str / dict / None) non-linear activation function
        to apply to the final layer.
        :param dropout: (float/ None) rate of dropout to apply to the FC layer,
        if None than doesn't apply dropout, default = None
        :param norm: (dict / None) Denotes the normalization layer to use with the
        FC layer. The dict should contains at least two keys, 'name' for indicating
        the type of normalization to use, and 'params', which should also map to a dict
        with all required parameters for the normalization layer. At the minimum,
        the 'params' dict should define the 'num_channels' key to indicate the expected
        number of channels on which to apply the normalization. For the GroupNorm,
        it is also required to specify a 'num_groups' key.
        If None then doesn't add normalization layer.
        Currently supported normalization layers: 'batch1d', 'batch2d', 'batch3d',
        'instance1d', 'instance2d', 'instance3d', 'group', where 'batch' stands for
        BatchNorm, `instance` stands for InstanceNorm and `group` stands
        for GroupNorm. Default == None.
        :param skip_connections: (bool) Whether to use skip connections,
        making the number of units to be l0_units in all layers
        """

        super(FCEncoderDecoder, self).__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim

        # Validate inputs
        assert ((units_per_layer is not None and units_factor is None) or
                units_per_layer is None and units_factor is not None and
                l0_units is not None), \
            f"Cannot specify both 'units_per_layer' = {units_per_layer} and" \
            f" 'units_factor' = {units_factor}, 'l0_units' = {l0_units}. " \
            "Please specify either 'units_per_layer' or " \
            "'units_factor' and 'l0_units'."

        if units_per_layer is not None:
            assert len(units_per_layer) == n_layers, \
                f"If 'units_per_layer' is not None, then it should specify the " \
                f"# units for every layer," \
                f" however {len(units_per_layer)} specification are" \
                f" given for {n_layers} layers."

            out_dim = units_per_layer[0]

        else:
            out_dim = l0_units

        if isinstance(norm, str) and norm in (
                'batch1d',
                'batch2d',
                'batch3d',
                'instance1d',
                'instance2d',
                'instance3d',
                'layer',
        ):
            norm = {
                'name': norm,
                'params': input_dim,
            }

        elif isinstance(norm, dict) and ('name' not in norm.keys() or
                                         'params' not in norm.keys()):
            raise ValueError(
                "If norm is a dict, it must contain the 'name' and 'params' keys."
            )

        elif norm is not None and not isinstance(norm, dict):
            raise ValueError(
                "norm must be either a string of: 'batch1d', 'batch2d', 'batch3d',"
                " 'instance1d', 'instance2d', 'instance3d', or None, or a dict"
            )

        # Build model
        layers = []
        if skip_connections:
            layers.append(
                get_fc_layer(
                    input_dim=input_dim,
                    output_dim=out_dim,
                    bias=bias,
                    activation=activation,
                    dropout=dropout,
                )
            )

        n_inner_layers = n_layers - 2 if skip_connections else n_layers - 1
        in_dim = out_dim if skip_connections else input_dim

        if norm is not None and norm['name'] == 'layer':
            norm['params']['normalized_shape'][-1] = out_dim

        if units_per_layer is not None and not skip_connections:
            out_dim = units_per_layer[1]

        for i in range(n_inner_layers):
            if skip_connections:
                layers.append(
                    SkipConnectionFCBlock(
                        input_dim=in_dim,
                        output_dim=out_dim,
                        bias=bias,
                        activation=activation,
                        dropout=dropout,
                        norm=norm,
                    )
                )

            else:
                layers.append(
                    get_fc_layer(
                        input_dim=in_dim,
                        output_dim=out_dim,
                        bias=bias,
                        activation=activation,
                        dropout=dropout,
                        norm=norm,
                    )
                )

            in_dim = out_dim
            if units_per_layer is not None and not skip_connections:
                out_dim = units_per_layer[i + 1]

            elif i > 0 and i % 2 == 0 and not skip_connections:
                out_dim = int(in_dim * units_factor)

            if norm is not None and norm['name'] == 'layer':
                norm['params']['normalized_shape'][-1] = out_dim

            elif norm is not None:
                norm = {
                    'params': in_dim,
                }

        layers.append(
            get_fc_layer(
                input_dim=in_dim,
                output_dim=output_dim,
                bias=bias,
                activation=final_activation,
            )
        )
        self._model = nn.Sequential(*layers)

        # Initialize weights
        self._model.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward logic for the 'FCEncoderDecoder' class.

        :param x: (Tensor) The input tensor.

        :return: (Tensor) The resulting tensor from the forward pass.
        """

        outputs = self._model(x)

        return outputs


class BaseMIDST(nn.Module):
    def __init__(
            self,
            m_dynamics: int,
            observable_dim: int = 64,
            k_forward_prediction: int = 1,
            eps: float = 1e-4,
            residual_dynamics: bool = False,
    ):
        super(BaseMIDST, self).__init__()

        self._m_dynamics = m_dynamics
        self._observable_dim = observable_dim
        self._k_forward_prediction = k_forward_prediction
        self._eps = eps
        self._residual_dynamics = residual_dynamics

        # Initialize dynamics
        U = [
            nn.init.xavier_normal_(
                torch.zeros(
                    (observable_dim, observable_dim)
                )
            ).type(torch.float32)
            for _ in range(k_forward_prediction)
        ]
        self._U_per_t = nn.ParameterList(
            [
                nn.parameter.Parameter(
                    u, requires_grad=True,
                )
                for u in U
            ]
        )
        V = [
            nn.init.xavier_normal_(
                torch.zeros(
                    (observable_dim, observable_dim)
                )
            ).type(torch.float32)
            for _ in range(k_forward_prediction)
        ]
        self._V_per_t = nn.ParameterList(
            [
                nn.parameter.Parameter(
                    v, requires_grad=True,
                )
                for v in V
            ]
        )
        S = [
            [
                nn.init.xavier_normal_(
                    torch.zeros(
                        (observable_dim, observable_dim)
                    )
                ).type(torch.float32)
                for _ in range(m_dynamics)
            ]
            for _ in range(k_forward_prediction)
        ]
        self._S_per_t_per_m = nn.ModuleList(
            [
                nn.ParameterList(
                    [
                        nn.parameter.Parameter(
                            S[t][m], requires_grad=True,
                        )
                        for m in range(m_dynamics)
                    ]
                )
                for t in range(k_forward_prediction)
            ]
        )

    def _get_sigmas(self) -> Union[nn.ModuleList, nn.ParameterList]:
        sigmas = self._S_per_t_per_m
        return sigmas

    def _get_u(self) -> Union[Sequence[Tensor], nn.ModuleList, nn.ParameterList]:
        return self._U_per_t

    def _get_v(self) -> Union[Sequence[Tensor], nn.ModuleList, nn.ParameterList]:
        return self._V_per_t

    def _get_koopmans(
            self,
    ) -> Sequence[Tensor]:
        s = self._get_sigmas()
        u = self._get_u()
        v = self._get_v()

        koopmans = [
            torch.cat(
                [
                    (
                        u[t].matmul(s[t][m]).matmul(
                            v[t].transpose(1, 0)
                        )
                    )[None, ...]
                    for m in range(self._m_dynamics)
                ],
                dim=0
            )
            for t in range(self._k_forward_prediction)
        ]

        return koopmans

    def _apply_koopmans(self, observables: Tensor) -> Tuple[Tensor, Sequence[Sequence[Tensor]]]:
        # If in residual mode than first compute the residuals
        if self._residual_dynamics:
            assert observables.shape[2] > 1, f"Cannot use residual mode with {observables.shape[2]=}"
            x = observables[..., 1:, :] - observables[..., :-1, :]

        else:
            x = observables

        # Unpack to each temporal element of each dynamic component in the batch
        observables_ = torch.reshape(
            x,
            ((x.shape[0] * x.shape[1]), x.shape[2], x.shape[3])
        )

        # Apply the evolution operators
        koopmans_per_system_per_t = self._get_koopmans()
        if self._m_dynamics == 1 and observables.shape[1] != 1:
            r = observables_.shape[0]

        else:
            r = observables_.shape[0] // koopmans_per_system_per_t[0].shape[0]

        if len(koopmans_per_system_per_t) == 1:
            observables_per_system_per_t = torch.reshape(
                bmm(
                    koopmans_per_system_per_t[0].repeat(r, 1, 1),
                    observables_.transpose(1, 2)
                ).transpose(1, 2),
                x.shape
            )

        else:
            observables_per_system_per_t = cat(
                [
                    torch.reshape(
                        bmm(
                            koopmans_t.repeat(r, 1, 1),
                            observables_.transpose(1, 2)
                        ).transpose(1, 2),
                        x.shape
                    )
                    for koopmans_t in koopmans_per_system_per_t
                ],
                dim=2
            )

        # If in residual mode than add back the initial values
        if self._residual_dynamics:
            if self._k_forward_prediction > 1:
                observables_per_system_per_t = torch.split(
                    (
                            observables[..., 0, :][..., None, :] + observables_per_system_per_t.cumsum(dim=2)
                    ),
                    x.shape[2],
                    dim=2,
                )
                observables_per_system_per_t = [
                    torch.cat([observables[..., 0, :][..., None, :], obs], dim=2)
                    for obs in observables_per_system_per_t
                ]
                observables_per_system_per_t = torch.cat(observables_per_system_per_t, dim=2)

            else:
                observables_per_system_per_t = cat(
                    (
                        observables[..., 0, :][..., None, :],
                        observables[..., 0, :][..., None, :] + observables_per_system_per_t.cumsum(dim=2),
                    ),
                    dim=2
                )

        return observables_per_system_per_t, koopmans_per_system_per_t

    def koopman_params(self) -> Sequence[torch.nn.Parameter]:
        params = []
        for u in self._U_per_t:
            params.append(u)

        for v in self._V_per_t:
            params.append(v)

        for p_module in self._S_per_t_per_m:
            for p in p_module:
                params.append(p)

        return params

    @abstractmethod
    def forward(self, x: Tensor) -> Dict:
        pass

    def __call__(self, x: Tensor) -> Dict:
        return self.forward(x)


class MIDST(BaseMIDST):
    def __init__(
            self,
            m_dynamics: int,
            states_dim: int,
            observable_dim: int = 64,
            n_encoder_layers: int = 8,
            l0_units: int = 1024,
            units_factor: float = 0.5,
            activation: Union[str, dict] = 'relu',
            final_activation: Union[str, dict] = None,
            norm: Union[dict, str] = None,
            dropout: float = 0.2,
            bias: bool = False,
            k_forward_prediction: int = 1,
            eps: float = 1e-4,
            residual_dynamics: bool = False,
            skip_connections: bool = True,
            use_revin: bool = True,
            revin_affine: bool = False,
            revin_remove_last: bool = True,
            non_stationary_norm: bool = False,
    ):
        super(MIDST, self).__init__(
            m_dynamics=m_dynamics,
            observable_dim=observable_dim,
            k_forward_prediction=k_forward_prediction,
            eps=eps,
            residual_dynamics=residual_dynamics,
        )

        self._states_dim = states_dim
        self._dropout = dropout
        self._use_revin = use_revin
        self._revin_affine = revin_affine
        self._revin_remove_last = revin_remove_last
        self._non_stationary_norm = non_stationary_norm

        if use_revin:
            self._revin = RevIN(m_dynamics, affine=revin_affine, subtract_last=revin_remove_last)

        if n_encoder_layers > 0:
            self.encoder = FCEncoderDecoder(
                input_dim=states_dim,
                output_dim=observable_dim,
                n_layers=n_encoder_layers,
                l0_units=l0_units,
                units_factor=units_factor,
                activation=activation,
                final_activation=final_activation,
                norm=norm,
                dropout=dropout,
                bias=bias,
                skip_connections=skip_connections,
            )
            self.decoder = FCEncoderDecoder(
                input_dim=observable_dim,
                output_dim=states_dim,
                n_layers=n_encoder_layers,
                l0_units=int(l0_units * (units_factor ** (n_encoder_layers - 1))),
                units_factor=(1 / units_factor),
                activation=activation,
                final_activation=final_activation,
                norm=norm,
                dropout=dropout,
                bias=bias,
                skip_connections=skip_connections,
            )

    def forward(self, states_t_0: Tensor) -> Dict[str, Tensor]:
        # Validate inputs
        assert len(states_t_0.shape) == 4, \
            f"The input tensor x must be a 4-dimensional Tensor, " \
            f"however states_t_0 is a " \
            f"{len(states_t_0.shape)}-dimensional Tensor."

        assert states_t_0.shape[1] == self._m_dynamics, \
            f"The model expects {self._m_dynamics} " \
            f"dynamics, however the input tensor states_t_0 has only " \
            f"{states_t_0.shape[1]} dynamics."

        assert states_t_0.shape[3] == self._states_dim, \
            f"The model expects {self._states_dim} observable components," \
            f" however the input tensor states_t_0 has {states_t_0.shape[3]} " \
            f"observable components."

        if self._non_stationary_norm:
            means = states_t_0.mean(dim=-2, keepdim=True).detach()
            states_t_0 = states_t_0 - means
            stdev = torch.sqrt(torch.var(states_t_0, dim=-2, keepdim=True, unbiased=False) + 1e-5)
            states_t_0 /= stdev

        if self._use_revin:
            states_t_0 = states_t_0.permute(0, 2, 3, 1)
            states_t_0 = self._revin(states_t_0, 'norm')
            states_t_0 = states_t_0.permute(0, 3, 1, 2)

        # Encode each row of observables
        observables_t_0 = self.encoder(states_t_0)

        # Apply the forward & inverse Koopman operators
        states_1, dynamics = self._apply_koopmans(observables_t_0)

        # Decode & reshape back to original shape
        states_t = self.decoder(states_1)
        reconstruction = self.decoder(observables_t_0)

        if self._use_revin:
            states_t = states_t.permute(0, 2, 3, 1)
            reconstruction = reconstruction.permute(0, 2, 3, 1)
            states_t = self._revin(states_t, 'denorm')
            reconstruction = self._revin(reconstruction, 'denorm')
            states_t = states_t.permute(0, 3, 1, 2)
            reconstruction = reconstruction.permute(0, 3, 1, 2)

        if self._non_stationary_norm:
            states_t = states_t * stdev
            states_t = states_t + means

            reconstruction = reconstruction * stdev
            reconstruction = reconstruction + means

        output = {
            MODELS_TENSOR_PREDICITONS_KEY: states_t,
            OTHER_KEY: {
                'states_0': observables_t_0,
                'states_1': states_1,
                'reconstruction': reconstruction,
                'dynamics': dynamics,
                'V_per_t': self._V_per_t,
                'U_per_t': self._U_per_t,
                'S_per_t': self._S_per_t_per_m,
            },
        }

        return output

    def nn_params(self) -> Sequence[torch.nn.Parameter]:
        params = list(self.encoder.parameters())
        params += list(self.decoder.parameters())
        return params
