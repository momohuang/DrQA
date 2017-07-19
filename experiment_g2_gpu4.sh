CUDA_VISIBLE_DEVICES=4 python train.py --do_C2Q --do_my_Q2C --name g2_gpu4 --save_best_only
CUDA_VISIBLE_DEVICES=4 python train.py --do_C2Q --do_my_Q2C --do_multi_att --multi_att_h 1 --multi_att_val 768 --multi_att_key 768 --name g2_gpu4 --save_best_only
CUDA_VISIBLE_DEVICES=4 python train.py --do_C2Q --do_coattention --int_ali_hidden_size 256 --name g2_gpu4 --save_best_only
