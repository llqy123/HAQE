cd .. 
source set_env.sh
python run.py \
            --dataset YAGO3-10 \
            --rank 32 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adam \
            --max_epochs 200 \
            --patience 20 \
            --valid 5 \
            --batch_size 500 \
            --neg_sample_size -1 \
            --init_size 0.001 \
            --learning_rate 0.001 \
            --gamma 0.0 \
            --bias learn \
            --dtype single \
            --multi_c
cd examples/
