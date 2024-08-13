from lldm import LOGS_DIR

import os
import glob
import pickle
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
exp_type = 'full_comparison'
# exp_type = 'illness_comparison'
# exp_type = 'ablate_window'

if exp_type == 'illness_comparison':
    exp_dirs = (
        'Illness',
    )
    exp_names = (
        'Illness',
    )

else:
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
# Only Without any normalization applied
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
lldm_4W_experiments = [
    sorted(
        glob.glob(
            os.path.join(
                logs_dir,
                exp_d,
                f'{exp_n}_LLDM*4W*'
            )
        )
    )
    for exp_d, exp_n in zip(exp_dirs, exp_names)
]
lldm_8W_experiments = [
    sorted(
        glob.glob(
            os.path.join(
                logs_dir,
                exp_d,
                f'{exp_n}_LLDM*8W*'
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

# --------------- MIDST ---------------
midst_experiments = [
    sorted(
        glob.glob(
            os.path.join(
                logs_dir,
                exp_d,
                f'{exp_n}_MIDST*'
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
# 12 Model
itransformer_12_experiments = [
    sorted(
        glob.glob(
            os.path.join(
                logs_dir,
                exp_d,
                f'{exp_n}_iTransformer*12H*'
            )
        )
    )
    for exp_d, exp_n in zip(exp_dirs, exp_names)
]

# 48 Model
itransformer_48_experiments = [
    sorted(
        glob.glob(
            os.path.join(
                logs_dir,
                exp_d,
                f'{exp_n}_iTransformer*48H*'
            )
        )
    )
    for exp_d, exp_n in zip(exp_dirs, exp_names)
]

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

save_file = os.path.join(LOGS_DIR, exp_type + '.pdf')
if __name__ == "__main__":
    if exp_type == "full_comparison":
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
        offsets = (
            0.01,
            -0.0075,
            -0.0125,
            0.0075,
            0.0075,
            0.01,
            -0.01,
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

    elif exp_type == "illness_comparison":
        experiments = (
            ptst_42_experiments,
            ptst_64_experiments,
            itransformer_12_experiments,
            itransformer_48_experiments,
            midst_experiments,
            lldm_2W_experiments,
            lldm_stationarity_norm_2W_experiments,
            lldm_revin_norm_2W_experiments,
        )
        legend = (
            'PatchTST-42',
            'PatchTST-64',
            'iTransformer-96',
            'iTransformer-720',
            'MIDST',
            'LLDM (Ours)',
            'LLDM+NStat (Ours)',
            'LLDM+RevIN (Ours)',
        )
        colors = (
            'springgreen',
            'darkgreen',
            'lightcoral',
            'firebrick',
            'darkviolet',
            'lightskyblue',
            'royalblue',
            'navy',
        )
        offsets = (
            0.025,
            -0.0075,
            -0.0125,
            0.0075,
            0.0075,
            0.025,
            -0.025,
        )
        markers = (
            'o',
            'x',
            'o',
            'x',
            'o',
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
            '-',
            '--',
            (0, (5, 5)),
        )

    elif exp_type == 'ablate_window':
        experiments = (
            lldm_2W_experiments,
            lldm_4W_experiments,
            lldm_8W_experiments,
        )
        legend = (
            'LLDM-2W',
            'LLDM-4W',
            'LLDM-8W',
        )
        colors = (
            'navy',
            'royalblue',
            'lightskyblue',
        )
        markers = (
            'o',
            'x',
            'd',
        )
        linestyles = (
            '-',
            '--',
            (0, (5, 5)),
        )

    test_accuracies = {
        name: {
            model: []
            for model in legend
        }
        for name in exp_names
    }
    for m, exp_per_model in enumerate(experiments):
        if exp_type != 'illness_comparison':
            # Re-Order to make it 96 | 192 | 336 | 720
            exp_per_model = [
                [
                    e
                    for e in ([exps[-1], ] + exps[:-1])
                ]
                for exps in exp_per_model
            ]
            x_axis = [96, 192, 336, 720]

        else:
            x_axis = [12, 24, 36, 48]

        for e, exp_data_type in enumerate(exp_per_model):
            for exp_per_horizon in exp_data_type:
                train_dir = os.path.join(exp_per_horizon, 'Train')
                val_dir = os.path.join(exp_per_horizon, 'Val')
                test_dir = os.path.join(exp_per_horizon, 'Test')
                test_acc_file = os.path.join(test_dir, 'eval_acc_1.pkl')

                with open(test_acc_file, 'rb') as f:
                    acc = pickle.load(f)
                    test_accuracies[exp_names[e]][legend[m]].append(acc[0])

    average_test_acc = {
        model_key: []
        for model_key in legend
    }
    print(f"{exp_type} Results:")
    avg_models_results = {
        model_key: [[], [], [], []]
        for model_key in legend
    }
    for data_i, data_key in enumerate(test_accuracies):
        for model_i, model_key in enumerate(test_accuracies[data_key]):
            if exp_type != 'ablate_window':
                print(f"{model_key}-{data_key} MSE: {test_accuracies[data_key][model_key]}")

            average_test_acc[model_key].append(np.array(test_accuracies[data_key][model_key])[None])
            avg_models_results[model_key][0].append(test_accuracies[data_key][model_key][0])
            avg_models_results[model_key][1].append(test_accuracies[data_key][model_key][1])
            avg_models_results[model_key][2].append(test_accuracies[data_key][model_key][2])
            avg_models_results[model_key][3].append(test_accuracies[data_key][model_key][3])

    avg_models_results = {
        model_key: [
            np.mean(avg_models_results[model_key][0][h])
            if len(avg_models_results[model_key][0]) > 1 else
            np.mean(avg_models_results[model_key][h])
            for h in range(4)
        ]
        for model_key in avg_models_results
    }
    average_test_acc = {
        model_key: np.concatenate(
            average_test_acc[model_key], axis=0
        ).mean(0)
        for model_key in legend
    }
    average_test_improvements = {
        model_key: (
                (
                        np.array(average_test_acc[model_key]).squeeze() -
                        np.array(average_test_acc['LLDM+RevIN (Ours)']).squeeze()
                ) / np.array(average_test_acc[model_key]).squeeze()
        ).mean()
        for model_key in average_test_acc
    }
    print(average_test_improvements)
    fig, ax = plt.subplots(figsize=[21, 11])
    for model_i, model_key in enumerate(legend):
        if exp_type == 'ablate_window':
            print(f"{model_key} MSE: {average_test_acc[model_key]}")

        ax.plot(
            x_axis,
            average_test_acc[model_key],
            color=colors[model_i],
            marker=markers[model_i],
            label=legend[model_i],
            linestyle=linestyles[model_i]
        )

    ax.set_ylabel(f'MSE', fontdict={'weight': 'bold'})
    ax.set_xlabel('Prediction Horizon', fontdict={'weight': 'bold'})
    ax.legend(prop={'weight': 'bold'})
    plt.savefig(
        fname=os.path.join(save_file),
        orientation='landscape',
        bbox_inches='tight',
        format='pdf',
    )
    plt.show()
