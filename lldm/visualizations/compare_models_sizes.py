from lldm import LOGS_DIR

import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

text_size = 20
plt.rcParams['axes.labelsize'] = text_size
plt.rcParams['axes.titlesize'] = text_size
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 15
plt.rcParams['xtick.labelsize'] = text_size
plt.rcParams['ytick.labelsize'] = text_size
plt.rcParams['font.size'] = text_size
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['legend.fontsize'] = text_size

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

# --------------- LLDM ---------------
lldm_2W_experiments = [
    sorted(
        glob.glob(
            os.path.join(
                logs_dir,
                exp_d,
                f'{exp_n}_LLDM*2W*'
            )
        )
    )
    for exp_d, exp_n in zip(exp_dirs, exp_names)
]

# Only Stationarity Norm applied
lldm_stationarity_norm_2W_experiments = [
    sorted(
        glob.glob(
            os.path.join(
                logs_dir,
                exp_d,
                f'{exp_n}_LLDM_NStat*2W*'
            )
        )
    )
    for exp_d, exp_n in zip(exp_dirs, exp_names)
]

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

save_file = os.path.join(LOGS_DIR, 'model_sizes.pdf')
if __name__ == "__main__":
    experiments = (
        ptst_42_experiments,
        ptst_64_experiments,
        itransformer_96_experiments,
        itransformer_720_experiments,
        lldm_2W_experiments,
        lldm_stationarity_norm_2W_experiments,
        lldm_revin_norm_2W_experiments,
    )
    legend = (
        'PatchTST-42',
        'PatchTST-64',
        'iTransformer-96',
        'iTransformer-720',
        'LLDM (Ours)',
        'LLDM+NStat (Ours)',
        'LLDM+RevIN (Ours)',
    )
    colors = (
        'springgreen',
        'darkgreen',
        'lightcoral',
        'firebrick',
        'lightskyblue',
        'royalblue',
        'navy',
    )
    markers = (
        'o',
        'x',
        'o',
        'x',
        'o',
        'x',
        'd',
    )
    linestyles = (
        '--',
        '-',
        '--',
        '-',
        '-',
        '--',
        (0, (5, 5)),
    )

    model_sizes = {
        name: {
            model: []
            for model in legend
        }
        for name in exp_names
    }
    x_axis = [96, 192, 336, 720]
    for m, exp_per_model in enumerate(experiments):
        # Re-Order to make it 96 | 192 | 336 | 720
        exp_per_model = [
            [
                e
                for e in ([exps[-1], ] + exps[:-1])
            ]
            for exps in exp_per_model
        ]

        for e, exp_data_type in enumerate(exp_per_model):
            for exp_per_horizon in exp_data_type:
                model_ckpt_path = os.path.join(exp_per_horizon, 'BestModel.PyTorchModule')
                model_ckpt = torch.load(model_ckpt_path)['model']
                n_params = sum(model_ckpt[p_name].numel() for p_name in model_ckpt)
                model_sizes[exp_names[e]][legend[m]].append(n_params)

    avg_models_sizes_per_horizon = {
        model_key: [[], [], [], []]
        for model_key in legend
    }
    for data_i, data_key in enumerate(model_sizes):
        for model_i, model_key in enumerate(model_sizes[data_key]):
            avg_models_sizes_per_horizon[model_key][0].append(model_sizes[data_key][model_key][0])
            avg_models_sizes_per_horizon[model_key][1].append(model_sizes[data_key][model_key][1])
            avg_models_sizes_per_horizon[model_key][2].append(model_sizes[data_key][model_key][2])
            avg_models_sizes_per_horizon[model_key][3].append(model_sizes[data_key][model_key][3])

    avg_models_sizes_per_horizon = {
        model_key: [
            np.mean(avg_models_sizes_per_horizon[model_key][h])
            for h in range(4)
        ]
        for model_key in avg_models_sizes_per_horizon
    }
    fig, ax = plt.subplots(figsize=[21, 11])
    for model_i, model_key in enumerate(legend):
        ax.plot(
            x_axis,
            avg_models_sizes_per_horizon[model_key],
            color=colors[model_i],
            marker=markers[model_i],
            label=legend[model_i],
            linestyle=linestyles[model_i]
        )

    ax.set_ylabel(f'# Trainable Parameters', fontdict={'weight': 'bold'})
    ax.set_xlabel('Prediction Horizon', fontdict={'weight': 'bold'})
    ax.legend(prop={'weight': 'bold'})
    plt.savefig(
        fname=os.path.join(save_file),
        orientation='landscape',
        bbox_inches='tight',
        format='pdf',
    )
    plt.show()
