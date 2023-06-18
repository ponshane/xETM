# linear orthogonal + MUSE dict + xETM
## airiti topic 50
CUDA_VISIBLE_DEVICES=1 python run.py \
    --data_path airiti_out \
    --train_embeddings 0 \
    --emb_path ./data/EMBED/full_corpus/100perc-en-zh-airiti-space.txt \
    --rho_size 100 \
    --emb_size 100 \
    --mode train \
    --epochs 30 | tee results_50.txt

# given pre-trained word embedding (only train topic embedding)
CUDA_VISIBLE_DEVICES=1 python run.py \
    --data_path airiti_out \
    --train_embeddings 0 \
    --emb_path ./data/EMBED/full_corpus/100perc-en-zh-airiti-space.txt \
    --rho_size 100 \
    --emb_size 100 \
    --load_from results/D_100_K_50_Epo_30_Opt_adam \
    --data_label_path ./college_label.csv \
    --shuffle_data_order_path ./doc_order6011.txt \
    --full_data_path ./data/training_all.txt \
    --suffix linear_orthogonal \
    --mode eval | tee experiment/topic_result_topic_50_D_100_full.txt

## supervised + custom dictionary
CUDA_VISIBLE_DEVICES=1 python run.py \
    --data_path airiti_out \
    --train_embeddings 0 \
    --emb_path ../MUSE/dumped/fasttext/2023-02-15/vectors-muse.txt \
    --rho_size 100 \
    --emb_size 100 \
    --mode train \
    --epochs 30 --save_path ./results/muse >results_muse_50.txt

# evaluate supervised MUSE + custom dict + xETM
CUDA_VISIBLE_DEVICES=1 python run.py \
    --data_path airiti_out \
    --train_embeddings 0 \
    --emb_path ../MUSE_back/dumped/fasttext/2023-02-15/vectors-muse.txt \
    --rho_size 100 \
    --emb_size 100 \
    --load_from ./results/muse/D_100_K_50_Epo_30_Opt_adam \
    --data_label_path ./college_label.csv \
    --shuffle_data_order_path ./doc_order6011.txt \
    --full_data_path ./data/training_all.txt \
    --mode eval | tee experiment/topic_result_topic_50_D_100_supervised_MUSE_custom_dict.txt

## GAN
CUDA_VISIBLE_DEVICES=1 python run.py \
    --data_path airiti_out \
    --train_embeddings 0 \
    --emb_path ../MUSE_back/vectors-muse-gan.txt \
    --rho_size 100 \
    --emb_size 100 \
    --batch_size 1024 \
    --mode train \
    --epochs 15 \
    --save_path ./results/muse_gan >results_muse_gan_50.txt

CUDA_VISIBLE_DEVICES=1 python run.py \
    --data_path airiti_out \
    --train_embeddings 0 \
    --emb_path ../MUSE_back/vectors-muse-gan.txt \
    --rho_size 100 \
    --emb_size 100 \
    --eval_batch_size 1024 \
    --load_from results/muse_gan/D_100_K_50_Epo_15_Opt_adam \
    --data_label_path ./college_label.csv \
    --shuffle_data_order_path ./doc_order6011.txt \
    --full_data_path ./data/training_all.txt \
    --suffix muse_gan \
    --mode eval | tee experiment/topic_result_topic_50_D_100_muse_gan.txt
