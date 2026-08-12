python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py \
--config ./configs/toru.yaml \
--checkpoint output/exp01/checkpoint_best.pth \
--evaluate

python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py \
--config ./configs/peter.yaml \
--checkpoint /home/lr/shenjl/elmo_home/sticker_chat_datasets/perceive_before_respond/output/exp01/checkpoint_best.pth \
--evaluate

# easy/hard
python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py \
--config ./configs/toru_hard.yaml \
--checkpoint output/exp01/checkpoint_best.pth \
--evaluate

python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py \
--config ./configs/toru_easy.yaml \
--checkpoint output/exp01/checkpoint_best.pth \
--evaluate

# val
python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py \
--config ./configs/toru_hard_val.yaml \
--checkpoint output/exp01/checkpoint_best.pth \
--evaluate

python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py \
--config ./configs/toru_easy_val.yaml \
--checkpoint output/exp01/checkpoint_best.pth \
--evaluate

python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py \
--config ./configs/toru_val.yaml \
--checkpoint output/exp01/checkpoint_best.pth \
--evaluate