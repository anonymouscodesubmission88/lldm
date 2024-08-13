from typing import Optional, Dict
from torch.utils.data import Dataset
from torch import Tensor, from_numpy, float32
from lldm.utils.defaults import GT_TENSOR_INPUTS_KEY, GT_TENSOR_PREDICITONS_KEY

import torch
import numpy as np
import pandas as pd


class BaseTSDataset(Dataset):
    def __init__(
            self,
            filepath: str,
            mode: str,
            step: int = 1,
            train_ratio: float = 0.6,
            val_ratio: float = 0.2,
            min_max_scale: bool = False,
            standard_scalar_scale: bool = True,
            invalid_value: Optional[float] = None,
            replacement_value: Optional[float] = None,
            remove_invalid_rows: bool = False,
    ):
        super(BaseTSDataset, self, ).__init__()

        assert mode in ('Train', 'Val', 'Test')
        assert (train_ratio + val_ratio) < 1
        assert not (min_max_scale and standard_scalar_scale)

        self._filepath = filepath
        self._step = step
        self._mode = mode
        self._min_max = min_max_scale
        self._standard_scalar = standard_scalar_scale
        self._invalid_value = invalid_value
        self._replacement_value = replacement_value
        self._remove_invalid_rows = remove_invalid_rows

        # Load data and handle invalid value if relevant
        data = pd.read_csv(filepath)

        # Filter out rows with invalid values
        if invalid_value is not None:
            if remove_invalid_rows:
                invalid_rows = []
                for col in data.columns:
                    invalid_rows.extend(list((data[col] == invalid_value).index))

                invalid_rows = np.unique(invalid_rows)
                data.drop(invalid_rows)

            else:
                data.replace(to_replace=invalid_value, value=replacement_value, inplace=True)

        # Load data
        self._columns = data.columns[1:]
        self._data = data[self._columns]

        dates = data[['date']]
        dates['date'] = pd.to_datetime(dates.date)
        dates['month'] = dates.date.apply(lambda row: row.month, 1)
        dates['day'] = dates.date.apply(lambda row: row.day, 1)
        dates['weekday'] = dates.date.apply(lambda row: row.weekday(), 1)
        dates['hour'] = dates.date.apply(lambda row: row.hour, 1)
        self._dates = dates.drop(labels=['date'], axis=1).values

        self._n_samples = len(self._data)
        self._n_train = int(train_ratio * self._n_samples)
        self._n_val = int(val_ratio * self._n_samples)
        self._n_test = self._n_samples - self._n_train - self._n_val

        if min_max_scale:
            self.scale = np.max(self._data.iloc[:self._n_train].values, axis=0, keepdims=True)
            self.bias = np.min(self._data.iloc[:self._n_train].values, axis=0, keepdims=True)

        elif standard_scalar_scale:
            self.scale = np.std(self._data.iloc[:self._n_train].values, axis=0, keepdims=True)
            self.bias = np.mean(self._data.iloc[:self._n_train].values, axis=0, keepdims=True)

        else:
            self.scale = 1
            self.bias = 0

        self._length = None

    def __len__(self) -> int:
        return self._length


class TSDataset(BaseTSDataset):

    def __init__(
            self,
            filepath: str,
            window_size: int,
            trajectory_length: int,
            mode: str,
            train_ratio: float = 0.6,
            val_ratio: float = 0.2,
            min_max_scale: bool = False,
            standard_scalar_scale: bool = True,
            invalid_value: Optional[float] = None,
            replacement_value: Optional[float] = None,
            remove_invalid_rows: bool = False,
            aug_noise_p: float = 0,
            aug_rescale_p: float = 0,
            aug_invert_p: float = 0,
            aug_permute_channels_p: float = 0,
            aug_permute_windows_p: float = 0,
            use_windows_of_states: bool = False,
    ):
        super(TSDataset, self).__init__(
            filepath=filepath,
            mode=mode,
            step=1,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            min_max_scale=min_max_scale,
            standard_scalar_scale=standard_scalar_scale,
            invalid_value=invalid_value,
            replacement_value=replacement_value,
            remove_invalid_rows=remove_invalid_rows,
        )

        self._window_size = window_size
        self._trajectory_length = trajectory_length
        self._sample_size = window_size * (trajectory_length + 1)
        self._aug_noise_p = aug_noise_p
        self._aug_rescale_p = aug_rescale_p
        self._aug_invert_p = aug_invert_p
        self._aug_permute_channels_p = aug_permute_channels_p
        self._aug_permute_windows_p = aug_permute_windows_p
        self._use_windows_of_states = use_windows_of_states

        receptive_field_size = window_size * trajectory_length

        if mode == "Train":
            self._start = 0
            n = self._n_train

        elif mode == 'Val':
            # For the first sample in the validation set, fill in the receptive field from the end of the training set
            self._start = self._n_train - receptive_field_size
            n = self._n_val + receptive_field_size

        else:
            # For the first sample in the test set, fill in the receptive field from the end of the validation set
            self._start = self._n_train + self._n_val - receptive_field_size
            n = self._n_test + receptive_field_size

        self._length = (n - self._sample_size)

    @staticmethod
    def _add_noise(
            x: np.ndarray,
            r: float = 0.05,
    ) -> np.ndarray:
        transformed_x = x + np.random.normal(0, (r * np.linalg.norm(x, 2, axis=-1, keepdims=True)), x.shape)
        return transformed_x

    @staticmethod
    def _invert(
            x: np.ndarray,
    ) -> np.ndarray:
        transformed_x = -x
        return transformed_x

    @staticmethod
    def _rescale(
            x: np.ndarray,
            scale: float = 0.1,
    ) -> np.ndarray:
        transformed_x = scale * x
        return transformed_x

    @staticmethod
    def _permute_channels(
            x: np.ndarray,
    ) -> np.ndarray:
        channels = np.arange(x.shape[0])
        new_channels_ordering = np.random.choice(channels, size=len(channels), replace=False)
        transformed_x = x[new_channels_ordering, ...]
        return transformed_x

    @staticmethod
    def _permute_windows(
            x: np.ndarray,
            y: np.ndarray,
    ) -> (np.ndarray, np.ndarray):
        windows = np.arange(x.shape[1])
        new_windows_ordering = np.random.choice(windows, size=len(windows), replace=False)
        transformed_x = x[:, new_windows_ordering, ...]
        transformed_y = y[:, new_windows_ordering, ...]
        return transformed_x, transformed_y

    def __getitem__(
            self,
            index: int,
    ) -> Dict[str, Tensor]:
        start = self._start + index
        data = self._data.iloc[start:(start + self._sample_size)].values
        data = (data - self.bias) / self.scale
        data = np.swapaxes(data, axis1=0, axis2=1)
        data = data.reshape((data.shape[0], (self._trajectory_length + 1), self._window_size))

        # SSL Augmentations
        u_noise = np.random.uniform()
        u_rescale = np.random.uniform()
        u_permute_c = np.random.uniform()
        u_permute_w = np.random.uniform()
        u_invert = np.random.uniform()
        if u_noise < self._aug_noise_p:
            r = np.random.uniform(0.01, 0.05)
            data = self._add_noise(data, r=r)
            x = data[:, :-1, :]
            y = data[:, 1:, :]

        elif u_rescale < self._aug_rescale_p:
            scale = np.random.uniform(0.5, 2)
            data = self._rescale(data, scale=scale)
            x = data[:, :-1, :]
            y = data[:, 1:, :]

        elif u_invert < self._aug_invert_p:
            data = self._invert(data)
            x = data[:, :-1, :]
            y = data[:, 1:, :]

        elif u_permute_c < self._aug_permute_channels_p:
            data = self._permute_channels(data)
            x = data[:, :-1, :]
            y = data[:, 1:, :]

        elif u_permute_w < self._aug_permute_windows_p:
            x = data[:, :-1, :]
            y = data[:, 1:, :]
            x, y = self._permute_windows(x=x, y=y)

        else:
            x = data[:, :-1, :]
            y = data[:, 1:, :]

            if self._use_windows_of_states:
                x = np.reshape(
                    np.swapaxes(np.swapaxes(x, 0, 1), 1, 2),
                    (1, (self._trajectory_length * self._window_size), -1),
                )
                y = np.reshape(
                    np.swapaxes(np.swapaxes(y, 0, 1), 1, 2),
                    (1, (self._trajectory_length * self._window_size), -1),
                )[:, -self._window_size:, :]

        assert not np.any(np.isnan(x))
        assert not np.any(np.isnan(y))

        sample = {
            GT_TENSOR_INPUTS_KEY: from_numpy(x).type(float32),
            GT_TENSOR_PREDICITONS_KEY: from_numpy(y).type(float32),
        }

        return sample


class TSFormersDataset(BaseTSDataset):
    def __init__(
            self,
            filepath: str,
            prediction_horizon: int,
            trajectory_length: int,
            labels_length: int,
            mode: str,
            train_ratio: float = 0.6,
            val_ratio: float = 0.2,
            min_max_scale: bool = False,
            standard_scalar_scale: bool = True,
            invalid_value: Optional[float] = None,
            replacement_value: Optional[float] = None,
            remove_invalid_rows: bool = False,
            swapaxes: bool = False,
    ):
        super(TSFormersDataset, self).__init__(
            filepath=filepath,
            mode=mode,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            min_max_scale=min_max_scale,
            standard_scalar_scale=standard_scalar_scale,
            invalid_value=invalid_value,
            replacement_value=replacement_value,
            remove_invalid_rows=remove_invalid_rows,
        )

        self._trajectory_length = trajectory_length
        self._labels_length = labels_length
        self._prediction_horizon = prediction_horizon
        self._sample_size = trajectory_length + prediction_horizon
        self._swapaxes = swapaxes

        if mode == "Train":
            self._start = 0
            n = self._n_train

        elif mode == 'Val':
            # For the first sample in the validation set, fill in the receptive field from the end of the training set
            self._start = self._n_train - trajectory_length
            n = self._n_val + trajectory_length

        else:
            # For the first sample in the test set, fill in the receptive field from the end of the validation set
            self._start = self._n_train + self._n_val - trajectory_length
            n = self._n_test + trajectory_length

        self._length = (n - self._sample_size)

    def __getitem__(
            self,
            index: int,
    ) -> Dict[str, Tensor]:
        start = self._start + index
        end = start + self._trajectory_length
        r_start = end
        r_end = r_start + self._prediction_horizon

        x = self._data.iloc[start:end].values
        y = self._data.iloc[r_start:r_end].values
        x = (x - self.bias) / self.scale
        y = (y - self.bias) / self.scale

        x_mark = self._dates[start:end]
        y_mark = self._dates[r_start:r_end]

        assert not np.any(np.isnan(x))
        assert not np.any(np.isnan(y))
        assert not np.any(np.isnan(x_mark))
        assert not np.any(np.isnan(y_mark))

        x = from_numpy(x).type(float32)
        y = from_numpy(y).type(float32)
        x_mark = from_numpy(x_mark).type(float32)
        y_mark = from_numpy(y_mark).type(float32)

        decoder_input = torch.zeros((self._prediction_horizon, y.shape[-1])).type(float32)
        decoder_input = torch.cat([x[-self._labels_length:, :], decoder_input], dim=0).type(float32)

        if self._swapaxes:
            x = x.permute(1, 0)
            y = y.permute(1, 0)

        sample = {
            GT_TENSOR_INPUTS_KEY: x,
            GT_TENSOR_PREDICITONS_KEY: y,
            'XMark': x_mark,
            'YMark': y_mark,
            'DecoderIn': decoder_input,
        }

        return sample
