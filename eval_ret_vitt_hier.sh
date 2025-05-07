conda activate HiCM2

python down_t5.py
SAVE_DIR="./presave/vitt"

python -m torch.distributed.launch --nproc_per_node 8 --master_port=29519 --use_env dvc_ret.py \
--bank_type vitt --window_size 30 --soft_k 30 --sim_match anchor_cos --sampling origin \
--save_dir=${SAVE_DIR} \
--load ./presave/vitt/best_model.pth --epochs=20 --lr=3e-4 \
--combine_datasets vitt --combine_datasets_val vitt --batch_size=2 --batch_size_val=2 --schedule="cosine_with_warmup" \
--ret_option hier_concat --hier_ret_num top-k --LLM_ver 70 --hier_use level_5 level_4 level_3 level_2 level_1 \
--eval
