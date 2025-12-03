python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py \
--config ./configs/tmp.yaml \
--output_dir ./output/exp01 \
--res_checkpoint /mnt/tsubasa/sticker_chat_datasets/perceive_before_respond/weights/128_res50/model_58_9149_6456.pth \
--vision_checkpoint /mnt/tsubasa/sticker_chat_datasets/perceive_before_respond/weights/ALBEF.pth