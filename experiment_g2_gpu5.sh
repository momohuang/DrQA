CUDA_VISIBLE_DEVICES=5 python train.py --do_C2Q --do_my_Q2C --name g2_gpu5 --save_best_only
CUDA_VISIBLE_DEVICES=5 python train.py --do_C2Q --do_my_Q2C --rnn_layers 2 --name g2_gpu5 --save_best_only
CUDA_VISIBLE_DEVICES=5 python train.py --do_coattention --name g2_gpu5 --save_best_only
