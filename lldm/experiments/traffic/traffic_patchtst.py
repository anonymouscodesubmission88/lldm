import os
import warnings

warnings.simplefilter(action='ignore')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from datetime import datetime
from lldm import DATA_DIR, LOGS_DIR
from lldm.utils.loggers import Logger
from lldm.utils.optim import Optimizer
from lldm.utils.trainers import Trainer
from lldm.utils.losses import ModuleLoss
from lldm.models.patchtst import PatchTST
from lldm.data.datasets import TSFormersDataset

import torch
import numpy as np

if __name__ == '__main__':
    # Set seed
    seed = 8783
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define the log dir
    prediction_horizon = 96
    patchtst_version = 42
    if patchtst_version == 64:
        trajectory_length = 512

    elif patchtst_version == 42:
        trajectory_length = 336

    date = str(datetime.today()).split()[0]
    fit_model = True
    description = f"{trajectory_length}W_" \
                  f"{prediction_horizon}H"

    experiment_name = f"Traffic_PatchTST{patchtst_version}_{description}_{date}"
    logs_dir = os.path.join(LOGS_DIR, experiment_name)
    os.makedirs(logs_dir, exist_ok=True)

    # Define the Datasets & Data loaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    num_workers = 4
    train_ratio = 0.7
    val_ratio = 0.1
    file_path = os.path.join(DATA_DIR, 'traffic', 'traffic.csv')
    standard_scalar_scale = True
    min_max_scale = False
    labels_length = (prediction_horizon // 2)
    train_ds = TSFormersDataset(
        filepath=file_path,
        prediction_horizon=prediction_horizon,
        trajectory_length=trajectory_length,
        labels_length=labels_length,
        mode='Train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        standard_scalar_scale=standard_scalar_scale,
        min_max_scale=min_max_scale,
    )
    val_ds = TSFormersDataset(
        filepath=file_path,
        prediction_horizon=prediction_horizon,
        trajectory_length=trajectory_length,
        labels_length=labels_length,
        mode='Val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        standard_scalar_scale=standard_scalar_scale,
        min_max_scale=min_max_scale,
    )
    test_ds = TSFormersDataset(
        filepath=file_path,
        prediction_horizon=prediction_horizon,
        trajectory_length=trajectory_length,
        labels_length=labels_length,
        mode='Test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        standard_scalar_scale=standard_scalar_scale,
        min_max_scale=min_max_scale,
    )

    pin_memory = True
    drop_last = False
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    # Define the model
    input_size = trajectory_length
    encoder_layers = 3
    n_heads = 16
    hidden_size = 128
    linear_hidden_size = 256
    dropout = 0.2
    fc_dropout = 0.2
    head_dropout = 0.0
    attn_dropout = 0.0
    patch_len = 16
    stride = 8
    revin = True
    revin_affine = False
    revin_subtract_last = True
    activation = "gelu"
    res_attention = True
    batch_normalization = False
    learn_pos_embed = True
    model_params = {
        'horizon': prediction_horizon,
        'input_size': input_size,
        'encoder_layers': encoder_layers,
        'n_heads': n_heads,
        'hidden_size': hidden_size,
        'linear_hidden_size': linear_hidden_size,
        'dropout': dropout,
        'fc_dropout': fc_dropout,
        'head_dropout': head_dropout,
        'attn_dropout': attn_dropout,
        'patch_len': patch_len,
        'stride': stride,
        'revin': revin,
        'revin_affine': revin_affine,
        'revin_subtract_last': revin_subtract_last,
        'activation': activation,
        'res_attention': res_attention,
        'batch_normalization': batch_normalization,
        'learn_pos_embed': learn_pos_embed,
    }
    model = PatchTST(
        **model_params
    )
    model.to(device)
    print(f"# Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Define the optimizer
    num_epochs = 100
    lr = 0.001
    weight_decay = 0.0001
    optimizer_hparams = {
        'lr': lr,
        'weight_decay': weight_decay,
    }
    optimizers = [
        torch.optim.AdamW(
            params=model.parameters(),
            **optimizer_hparams,
        ),
    ]

    scheduler_hparams = {
        'mode': 'min',
        'factor': 0.5,
        'patience': 10,
        'threshold': 1e-4,
        'threshold_mode': 'rel',
        'cooldown': 0,
        'min_lr': 1e-6,
        'eps': 1e-8,
        'verbose': True,
    }
    schedulers = [
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizers[0],
            **scheduler_hparams
        ),
    ]
    optimizer = Optimizer(optimizers=optimizers, schedulers=schedulers)

    loss_fn = ModuleLoss(
        model=torch.nn.MSELoss(),
    )
    evaluation_metric = ModuleLoss(
        model=torch.nn.MSELoss(),
    )

    # Define the logger
    logger = Logger(
        log_dir=LOGS_DIR,
        experiment_name=experiment_name,
        max_elements=2,
    )

    # Define the trainer
    checkpoints = True
    early_stopping = None
    checkpoints_mode = 'min'
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        evaluation_metric=evaluation_metric,
        optimizer=optimizer,
        device=device,
        logger=logger,
    )

    # Write Scenario Specs
    specs = {
        'Data Specs': '',
        "seed": seed,
        'prediction_horizon': prediction_horizon,
        'DataLoader Specs': '',
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
        'Model Specs': '',
        'Model': type(model).__name__,
    }
    specs.update(model_params)
    loss_params = {
        'Loss Specs': '',
        'loss_fn': f"{loss_fn}",
        'eval_fn': f"{evaluation_metric}",
        'Trainer Specs': '',
        'num_epochs': num_epochs,
        'checkpoints': checkpoints,
        'early_stopping': early_stopping,
        'checkpoints_mode': checkpoints_mode,
        'Optimizer Specs': '',
        'optimizer': type(optimizers[0]).__name__,
    }
    specs.update(loss_params)
    specs.update(optimizer_hparams)
    specs['LR Scheduler Specs'] = ''
    specs['lr_scheduler'] = type(schedulers[0]).__name__
    specs.update(scheduler_hparams)

    specs_file = os.path.join(logs_dir, 'data_specs.txt')
    with open(specs_file, 'w') as f:
        for k, v in specs.items():
            f.write(f"{k}: {str(v)}\n")

    print("Fitting the model")
    if fit_model:
        trainer.fit(
            dl_train=train_dl,
            dl_val=val_dl,
            num_epochs=num_epochs,
            checkpoints=checkpoints,
            checkpoints_mode=checkpoints_mode,
            early_stopping=early_stopping,
        )

    # Define the test-set
    print("Evaluating over the test set")
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    model = PatchTST(
        **model_params
    )
    model_ckpt_path = f"{logs_dir}/BestModel.PyTorchModule"  # loading best model
    model_ckp = torch.load(model_ckpt_path)
    model.load_state_dict(model_ckp['model'])
    model.to(device)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        evaluation_metric=evaluation_metric,
        optimizer=optimizer,
        device=device,
        logger=logger,
    )

    # Evaluate
    trainer.evaluate(
        dl_test=test_dl,
        ignore_cap=True,
    )
