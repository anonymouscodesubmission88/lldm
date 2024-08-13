from lldm import LOGS_DIR
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.projections.polar import PolarAxes
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.projections import register_projection

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

text_size = 24
plt.rcParams['axes.labelsize'] = text_size
plt.rcParams['axes.titlesize'] = text_size
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.markersize'] = 10
plt.rcParams['xtick.labelsize'] = text_size
plt.rcParams['ytick.labelsize'] = text_size
plt.rcParams['font.size'] = text_size
plt.rcParams['legend.fontsize'] = text_size
plt.rcParams['font.weight'] = 'bold'


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


logs_dir = LOGS_DIR
exp_dirs = (
    'ETT',
    'ETT',
    'ETT',
    'ETT',
    'Electricity',
    'Traffic',
    'Weather',
)
exp_names = (
    'ETT_H1',
    'ETT_H2',
    'ETT_M1',
    'ETT_M2',
    'Electricity',
    'Traffic',
    'Weather',
)
n_experiments = len(exp_names)

# --------------- MIDST ---------------
# Both RevIN & Stationarity Norms applied
midsts_2W_experiments = [
    sorted(
        glob.glob(
            os.path.join(
                logs_dir,
                exp_d,
                f'{exp_n}_MIDST_Residual*2W*'
            )
        )
    )
    for exp_d, exp_n in zip(exp_dirs, exp_names)
]

# --------------- LLDM ---------------
# Only RevIN Norm applied
lldm_revin_norm_2W_experiments = [
    sorted(
        glob.glob(
            os.path.join(
                logs_dir,
                exp_d,
                f'{exp_n}_LLDM_RevIN*2W*'
            )
        )
    )
    for exp_d, exp_n in zip(exp_dirs, exp_names)
]

# --------------- PatchTST ---------------
# 42 Model
ptst_42_experiments = [
    sorted(
        glob.glob(
            os.path.join(
                logs_dir,
                exp_d,
                f'{exp_n}_PatchTST42*'
            )
        )
    )
    for exp_d, exp_n in zip(exp_dirs, exp_names)
]

# 64 Model
ptst_64_experiments = [
    sorted(
        glob.glob(
            os.path.join(
                logs_dir,
                exp_d,
                f'{exp_n}_PatchTST64*'
            )
        )
    )
    for exp_d, exp_n in zip(exp_dirs, exp_names)
]

# --------------- iTransformer ---------------
# 96 Model
itransformer_96_experiments = [
    sorted(
        glob.glob(
            os.path.join(
                logs_dir,
                exp_d,
                f'{exp_n}_iTransformer*96H*'
            )
        )
    )
    for exp_d, exp_n in zip(exp_dirs, exp_names)
]

# 720 Model
itransformer_720_experiments = [
    sorted(
        glob.glob(
            os.path.join(
                logs_dir,
                exp_d,
                f'{exp_n}_iTransformer*720H*'
            )
        )
    )
    for exp_d, exp_n in zip(exp_dirs, exp_names)
]

fill_radar_plots = False
exp_type = 'comparison_radar'
save_file = os.path.join(LOGS_DIR, exp_type + '.pdf')
if __name__ == "__main__":
    experiments = (
        ptst_42_experiments,
        ptst_64_experiments,
        itransformer_96_experiments,
        itransformer_720_experiments,
        lldm_revin_norm_2W_experiments,
    )
    legend = (
        'PatchTST-42',
        'PatchTST-64',
        'iTransformer-96',
        'iTransformer-720',
        'LLDM+RevIN (Ours)',
    )
    colors = {
        'PatchTST-42': 'springgreen',
        'PatchTST-64': 'darkgreen',
        'iTransformer-96': 'lightcoral',
        'iTransformer-720': 'firebrick',
        'LLDM+RevIN (Ours)': 'royalblue',
    }
    theta = radar_factory(n_experiments, frame='polygon')

    test_accuracies = {
        name: {
            model: []
            for model in legend
        }
        for name in exp_names
    }
    for m, exp_per_model in enumerate(experiments):
        exp_per_model = [
            [
                e
                for e in ([exps[-1], ] + exps[:-1])
            ]
            for exps in exp_per_model
        ]
        for e, exp_type in enumerate(exp_per_model):
            for exp_per_horizon in exp_type:
                train_dir = os.path.join(exp_per_horizon, 'Train')
                val_dir = os.path.join(exp_per_horizon, 'Val')
                test_dir = os.path.join(exp_per_horizon, 'Test')
                test_acc_file = os.path.join(test_dir, 'eval_acc_1.pkl')
                with open(test_acc_file, 'rb') as f:
                    acc = pickle.load(f)
                    test_accuracies[exp_names[e]][legend[m]].append(acc[0])

    test_accuracies = {
        name: {
            n: test_accuracies[n][name]
            for n in exp_names
        }
        for name in legend
    }
    x_axis = [96, 192, 336, 720]
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=[32, 21],
        subplot_kw=dict(projection='radar'),
    )
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    axes[0][0].set_title(
        x_axis[0],
        weight='bold',
        position=(0.5, 1.1),
        horizontalalignment='center',
        verticalalignment='center',
    )
    axes[0][1].set_title(
        x_axis[1],
        weight='bold',
        position=(0.5, 1.1),
        horizontalalignment='center',
        verticalalignment='center',
    )
    axes[1][0].set_title(
        x_axis[2],
        weight='bold',
        position=(0.5, 1.1),
        horizontalalignment='center',
        verticalalignment='center',
    )
    axes[1][1].set_title(
        x_axis[3],
        weight='bold',
        position=(0.5, 1.1),
        horizontalalignment='center',
        verticalalignment='center',
    )

    grid = [0, 0.2, 0.4, 0.6, 0.8]
    labels = ['0 %', '20 %', '40 %', '60 %', '80 %']
    angle = -120
    axes[0][0].set_rgrids(radii=grid, labels=labels, angle=angle)
    axes[0][1].set_rgrids(radii=grid, labels=labels, angle=angle)
    axes[1][0].set_rgrids(radii=grid, labels=labels, angle=angle)
    axes[1][1].set_rgrids(radii=grid, labels=labels, angle=angle)
    max_values = [
        max(
            [
                test_accuracies[model_key][data_key][h]
                for model_key in test_accuracies
                for data_key in exp_names
            ]
        )
        for h in range(4)
    ]
    for i, model_name in enumerate(test_accuracies):
        model_data = test_accuracies[model_name]
        model_96_accs = [
            test_accuracies[model_name][exp_name][0]
            for exp_name in exp_names
        ]
        model_96_accs = (max_values[0] - np.array(model_96_accs)) / max_values[0]

        model_192_accs = [
            test_accuracies[model_name][exp_name][1]
            for exp_name in exp_names
        ]
        model_192_accs = (max_values[1] - np.array(model_192_accs)) / max_values[1]

        model_336_accs = [
            test_accuracies[model_name][exp_name][2]
            for exp_name in exp_names
        ]
        model_336_accs = (max_values[2] - np.array(model_336_accs)) / max_values[2]

        model_720_accs = [
            test_accuracies[model_name][exp_name][3]
            for exp_name in exp_names
        ]
        model_720_accs = (max_values[3] - np.array(model_720_accs)) / max_values[3]

        axes[0][0].plot(theta, model_96_accs, color=colors[model_name], label=model_name)
        axes[0][1].plot(theta, model_192_accs, color=colors[model_name], label=model_name)
        axes[1][0].plot(theta, model_336_accs, color=colors[model_name], label=model_name)
        axes[1][1].plot(theta, model_720_accs, color=colors[model_name], label=model_name)

        if fill_radar_plots:
            axes[0][0].fill(theta, model_96_accs, facecolor=colors[model_name], alpha=0.25)
            axes[0][1].fill(theta, model_192_accs, facecolor=colors[model_name], alpha=0.25)
            axes[1][0].fill(theta, model_336_accs, facecolor=colors[model_name], alpha=0.25)
            axes[1][1].fill(theta, model_720_accs, facecolor=colors[model_name], alpha=0.25)

    axes[0][0].set_varlabels(exp_names)
    axes[1][0].set_varlabels(exp_names)
    axes[0][1].set_varlabels(exp_names)
    axes[1][1].set_varlabels(exp_names)
    legend = axes[0, 0].legend(
        loc=(0.9, .95),
        labelspacing=0.1,
        prop={'weight': 'bold'}
    )
    plt.savefig(
        fname=os.path.join(save_file),
        orientation='landscape',
        bbox_inches='tight',
        format='pdf',
    )
    plt.show()
