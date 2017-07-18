CUDA_VISIBLE_DEVICES=5 python train.py --do_C2Q --gated_int_ali_doc --name g2_gpu5 --save_best_only
CUDA_VISIBLE_DEVICES=5 python train.py --do_C2Q --do_coattention --gated_int_ali_doc --name g2_gpu5 --save_best_only
CUDA_VISIBLE_DEVICES=5 python train.py --do_C2Q --doc_layers 4 --question_layers 4 --name g2_gpu5 --save_best_only
CUDA_VISIBLE_DEVICES=5 python train.py --do_C2Q --hidden_size 192 --name g2_gpu5 --save_best_only
