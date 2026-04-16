set -x

MASTER_ADDR=127.0.0.3
MASTER_PORT=11312
NUM_PROCESS=$1

cd $(dirname $0)

extra_kwargs=""
if [ $NUM_PROCESS -gt 1 ]; then
  distributed_type=MULTI_GPU
  extra_kwargs="${extra_kwargs}--multi_gpu"
else
  distributed_type=NO
fi

accelerate launch --config_file single_gpu.json \
  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT ${extra_kwargs} \
  --num_machines 1 --num_processes $NUM_PROCESS \
  accelerate_train_example.py \
  --train_batch_size=5 \
  --dataloader_num_workers=8 \
  --learning_rate=5e-05 \
  --weight_decay=0.01 \
  --checkpoint_root=${HOME}/train_outputs/checkpoint/debug/example \
  --show_root=${HOME}/train_outputs/show/debug/example \
  --save_steps=10 \
  --test_steps=20 \
  --train_visual_steps=5 \
  --train_steps=50 \
  --max_local_state_num=2 \
  ${@:2}


# /home/tiger/code/unhcv/unhcv/projects/diffusion/ip_adapter/accelerate_config_single_gpu.json
# --hdfs_root=hdfs://haruna/dp/mloops/datasets/zyx_data/checkpoints/unet_inpainting \
# --extra_checkpoint=/home/tiger/checkpoint/ip_adapter_tuchong/checkpoint-30000/unet/model.bin
