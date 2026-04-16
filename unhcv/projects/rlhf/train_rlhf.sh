set -x
ulimit -n 4096

extra_kwargs=""
REMAINING_ARGS=()

# 允许设置的变量列表
train_batch_size=1; dataloader_num_workers=4; learning_rate=1e-05; weight_decay=0.01;
save_steps=100; test_steps=10000000; train_visual_steps=100; train_steps=800; max_local_state_num=1; mixed_precision=bf16;
max_grad_norm=0;
launch_config_file="deep_speed_bf16.yml"
launch_config_file="single_gpu.json"
#launch_config_file="accelerate_config.json"

ALLOWED_KEYS=(train_batch_size dataloader_num_workers learning_rate weight_decay save_steps test_steps \
train_visual_steps train_steps max_local_state_num mixed_precision launch_config_file max_grad_norm)

set +x
# 检查 key 是否在白名单内
is_allowed_key() {
  local input_key="$1"
  for allowed in "${ALLOWED_KEYS[@]}"; do
    [[ "$allowed" == "$input_key" ]] && return 0
  done
  return 1
}

# 解析参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --*)
      key="${1/--/}"
      shift
      # 收集所有后续非选项参数作为值
      values=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        values+=("$1")
        shift
      done
      if is_allowed_key "$key"; then
          declare "$key=${values[*]}"
        else
          REMAINING_ARGS+=("--$key")
          REMAINING_ARGS+=("${values[@]}")
      fi
      ;;
    *)
      REMAINING_ARGS+=("$1")
      shift
      ;;
  esac
done
set -x

###<editor-fold desc="4090 gpu check">
set +x
gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader)

# 查找是否包含 4090
if echo "$gpu_info" | grep -i "4090" > /dev/null; then
    echo "✅ RTX 4090 GPU detected."
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
else
    echo "❌ No RTX 4090 found."
fi
set -x
###</editor-fold>

###<editor-fold desc="check distributed type">
MASTER_ADDR=127.0.0.3
MASTER_PORT=11312
MASTER_PORT=$((RANDOM % 100 + ${MASTER_PORT}))
NUM_PROCESS=${REMAINING_ARGS[0]}

cd $(dirname $0)

#if [ $NUM_PROCESS -gt 1 ]; then
#  extra_kwargs="${extra_kwargs}--multi_gpu"
#fi
echo $extra_kwargs
###</editor-fold>

accelerate launch --config_file ${HOME}/code/unhcv_refactor/unhcv/core/train/${launch_config_file} \
  --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT ${extra_kwargs} \
  --num_machines 1 --num_processes $NUM_PROCESS --mixed_precision $mixed_precision \
  train_rlhf.py \
  --train_batch_size=$train_batch_size \
  --dataloader_num_workers=$dataloader_num_workers \
  --learning_rate=$learning_rate \
  --weight_decay=$weight_decay \
  --checkpoint_root=${HOME}/train_outputs/checkpoint/rlhf \
  --show_root=${HOME}/train_outputs/show/rlhf \
  --save_steps=$save_steps \
  --test_steps=$test_steps \
  --train_visual_steps=$train_visual_steps \
  --train_steps=$train_steps \
  --max_local_state_num=$max_local_state_num \
  --max_grad_norm=$max_grad_norm \
  --metric_collect_file=train_outputs/show/rlhf/metric_collect/metric.yml \
  "${REMAINING_ARGS[@]:1}"

