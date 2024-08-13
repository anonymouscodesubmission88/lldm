from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn, Tensor
from abc import ABC, abstractmethod
from typing import Sequence, Optional, Union, Dict
from lldm.utils.defaults import (
    MODELS_TENSOR_PREDICITONS_KEY,
    GT_TENSOR_PREDICITONS_KEY,
)


class LossComponent(nn.Module, ABC):
    """
    An abstract API class for loss models
    """

    def __init__(self):
        super(LossComponent, self).__init__()

    @abstractmethod
    def forward(self, inputs: Union[Dict, Sequence]) -> Tensor:
        """
        The forward logic of the loss class.

        :param inputs: (Dict) Either a dictionary with the predictions from the forward pass and the ground truth
        outputs. The possible keys are specified by the following variables:
            MODELS_TENSOR_PREDICITONS_KEY
            MODELS_SEQUENCE_PREDICITONS_KEY
            GT_TENSOR_PREDICITONS_KEY
            GT_SEQUENCE_PREDICITONS_KEY
            GT_TENSOR_INPUTS_KEY
            GT_SEQUENCE_INPUTS_KEY
            OTHER_KEY

        Which can be found under .../DynamicalSystems/utils/defaults.py

        Or a Sequence of dicts, each where each element is a dict with the above-mentioned structure.

        :return: (Tensor) A scalar loss.
        """

        raise NotImplemented

    def __call__(self, inputs: Union[Dict, Sequence]) -> Tensor:
        return self.forward(inputs=inputs)

    def update(self, params: Dict[str, any]):
        for key in params:
            if hasattr(self, key):
                setattr(self, key, params[key])


class ModuleLoss(LossComponent):
    """
    A LossComponent which takes in a PyTorch loss Module and decompose the inputs according to the module's
    expected API.
    """

    def __init__(
            self, model: nn.Module,
            scale: float = 1.0,
            last_window_only: bool = False,
            last_window_pred_only: bool = False,
            trajectory_length: Optional[int] = None,
    ):
        """
        The constructor for the ModuleLoss class
        :param model: (PyTorch Module) The loss model, containing the computation logic.
        :param scale: (float) Scaling factor for the loss.
        """

        super().__init__()

        self.model = model
        self.scale = scale
        self._last_window_only = last_window_only
        self._last_window_pred_only = last_window_pred_only
        self._trajectory_length = trajectory_length

    def forward(self, inputs: Dict) -> Tensor:
        """
        Basically a wrapper around the forward of the inner model, which decompose the inputs to the expected
        structure expected by the PyTorch module.

        :param inputs: (dict) The outputs of the forward pass of the model along with the ground-truth labels.
        :return: (Tensor) A scalar Tensor representing the aggregated loss
        """

        y_pred = inputs[MODELS_TENSOR_PREDICITONS_KEY]
        y = inputs[GT_TENSOR_PREDICITONS_KEY]

        if self._last_window_only:
            y = y[..., -1, :]
            y_pred = y_pred[..., -1, :]

        if self._last_window_pred_only:
            y_pred = y_pred[..., (self._trajectory_length - 1)::self._trajectory_length, :]

        loss = self.scale * self.model(y_pred, y)

        return loss
