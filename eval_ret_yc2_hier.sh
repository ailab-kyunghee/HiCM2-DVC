conda activate HiCM2

SAVE_DIR="./presave/yc2"
python down_t5.py

python -m torch.distributed.launch --nproc_per_node 1 --master_port=29111 --use_env dvc_ret.py \
--bank_type yc2 --window_size 10 --sim_match anchor_cos --sampling origin \
--save_dir=${SAVE_DIR} --load ./presave/yc2/best_model.pth --epochs=20 --lr=3e-4 \
--combine_datasets youcook --combine_datasets_val youcook --batch_size=2 --batch_size_val=2 --schedule="cosine_with_warmup" \
--ret_option hier_concat --hier_ret_num top-k --soft_k 10 --LLM_ver 70 --hier_use level_4 level_3 level_2 level_1 \
--eval
