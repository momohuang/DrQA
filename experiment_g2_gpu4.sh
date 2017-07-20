CUDA_VISIBLE_DEVICES=4 python train.py --tune_partial 100 --name g2_gpu4 --save_best_only
CUDA_VISIBLE_DEVICES=4 python train.py --do_C2Q --do_my_Q2C --name g2_gpu4 --save_best_only
CUDA_VISIBLE_DEVICES=4 python train.py --do_coattention --rnn_layers 2 --name g2_gpu4 --save_best_only
