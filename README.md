# ROBOT

This repo contains the sample code of our proposed framework Slack Federated Adversarial Training (SFAT) in our paper: Combating Exacerbated Heterogeneity for Robust Models in Federated Learning (ICLR 2023).

Run the code with the following commands:
```
python robot_main.py --dataset cifar10 --corruption_ratio 0.5 --analyze --meta_lr 5e-2 --wandb --start_correction 0 --runs_name 0.45_forward_rce_uniform --loss forward --print_predictions --start_updating_T 20 --max_epoch 85 --outer_obj rce --corruption_type uniform

python robot_main.py --dataset cifar10 --corruption_ratio 0.45 --analyze --meta_lr 5e-2 --wandb --start_correction 0 --runs_name 0.45_forward_rce_flip --loss forward --print_predictions --start_updating_T 20 --max_epoch 85 --outer_obj rce --corruption_type flip1

python robot_main.py --dataset cifar100 --corruption_ratio 0.2 --analyze --meta_lr 5e-2 --start_correction 0 --loss forward --start_updating_T 20 --lr 0.05 --weight_decay 1e-3 --momentum 0.9 --T_init 4.5 --batch_size 128 --meta_optim adam --project rce_T_revision_cifar100_partial_inner0.05 --runs_name 0.2_outlr5e-2_adam_mae --wandb --network r34 --num_meta 0 --max_epoch 200 --outer_obj mae --corruption_type flip1 --meta_batch_size 512

python robot_main.py  --dataset cifar100 --corruption_ratio 0.45 --analyze --meta_lr 1e-2 --start_correction 0 --loss forward --start_updating_T 10 --lr 0.05 --weight_decay 1e-3 --momentum 0.9 --T_init 4.5 --batch_size 128 --meta_optim adam --project rce_T_revision_cifar100_partial_inner0.05 --runs_name 0.45_outlr3e-3_adam_mae --wandb --network r34 --num_meta 0 --max_epoch 200 --outer_obj mae --corruption_type flip1 --meta_batch_size 512
```
