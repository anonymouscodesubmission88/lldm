from typing import Union
from lldm.models.midst import MIDST


class LLDM(MIDST):
    def __init__(
            self,
            states_dim: int,
            prediction_horizon: int = 1,
            latent_dimensions: int = 64,
            n_encoder_layers: int = 8,
            l0_units: int = 1024,
            units_factor: float = 0.5,
            activation: Union[str, dict] = 'relu',
            final_activation: Union[str, dict] = None,
            norm: Union[dict, str] = None,
            dropout: float = 0.2,
            bias: bool = False,
            use_revin: bool = True,
            revin_affine: bool = False,
            revin_remove_last: bool = True,
            non_stationary_norm: bool = False,
    ):
        super(LLDM, self).__init__(
            m_dynamics=states_dim,
            states_dim=prediction_horizon,
            observable_dim=latent_dimensions,
            n_encoder_layers=n_encoder_layers,
            l0_units=l0_units,
            units_factor=units_factor,
            activation=activation,
            final_activation=final_activation,
            norm=norm,
            dropout=dropout,
            bias=bias,
            k_forward_prediction=1,
            residual_dynamics=True,
            use_revin=use_revin,
            revin_affine=revin_affine,
            revin_remove_last=revin_remove_last,
            non_stationary_norm=non_stationary_norm,
        )
