
# # 20250219 TransformerDualClassifier
# # Did not finished, stop manually, try out sweep from wandb
# for lr in 0.0001 0.0005 0.001 0.005; do
#     for num_heads in 2 4 8; do
#         for num_layers in 2 4 8; do
#             for dropout in 0.1 0.3 0.5 0.7; do
#                 for batch_size in 16 32 64 128; do
#                     for model_dim in 64 128 256 512; do
#                         python3 project/train.py \
#                             --model_name TransformerDualClassifier \
#                             --learning_rate $lr \
#                             --scheduler ExponentialLR \
#                             --gamma 0.95 \
#                             --batch_size $batch_size \
#                             --epochs 300 \
#                             --model_dim $model_dim \
#                             --num_heads $num_heads \
#                             --num_layers $num_layers \
#                             --dropout $dropout
#                     done
#                 done
#             done
#         done
#     done
# done

# ### 20250217
# # test sequence length = pad max length
# lr=0.001
# num_filters=200
# output_size=128
# dropout=0.3
# batch_size=64
# python3 project/train.py \
#     --model_name ACNN \
#     --learning_rate $lr \
#     --scheduler ExponentialLR \
#     --gamma 0.95 \
#     --batch_size $batch_size \
#     --epochs 300 \
#     --num_filters $num_filters \
#     --output_size $output_size \
#     --dropout $dropout \
#     --max_length 530

### 20250216
# test all
# for lr in 0.0001 0.0005 0.001 0.005 0.01 0.05; do
#     for num_filters in 100 200 300; do
#         for dropout in 0.3 0.5 0.7; do
#             for output_size in 128 256 512; do
#                 for batch_size in 16 32 64 128; do
#                     python3 project/train.py \
#                         --model_name ACNN \
#                         --learning_rate $lr \
#                         --scheduler ExponentialLR \
#                         --gamma 0.95 \
#                         --batch_size $batch_size \
#                         --epochs 300 \
#                         --num_filters $num_filters \
#                         --output_size $output_size \
#                         --dropout $dropout
#                 done
#             done
#         done
#     done
# done

### 20250216
# test lr schedular: Exponential LR: gamma=0.95

# num_filters=200
# dropout=0.3
# for lr in 0.0001 0.0005 0.001 0.005; do
#     python3 project/train.py \
#         --model_name ACNN \
#         --learning_rate $lr \
#         --scheduler ExponentialLR \
#         --gamma 0.95 \
#         --batch_size 32 \
#         --epochs 100 \
#         --num_filters $num_filters \
#         --output_size 128 \
#         --dropout $dropout
#     done

### 20250212
# test architecture num_filters 100 -> 200
# for num_filters in 100 200 300; do
#     for lr in 0.0001 0.0005 0.001 0.005; do
#         for dropout in 0.3 0.5 0.7; do
#             python3 project/train.py \
#                 --model_name ACNN \
#                 --learning_rate $lr \
#                 --batch_size 32 \
#                 --epochs 100 \
#                 --num_filters $num_filters \
#                 --output_size 128 \
#                 --dropout $dropout
#         done
#     done
# done

# find finished 36 runs
# find . -type d -name "run-20250212*" -newermt "2025-02-12 08:44" | wc -l


### 20250211
# # tune lr
# # lr=0.005 yield the best: 25.9% accuracy
# for lr in 0.0001 0.0005 0.001 0.005; do
# python3 project/train.py \
#     --model_name ACNN \
#     --learning_rate $lr \
#     --batch_size 32 \
#     --epochs 100 \
#     --num_filters 100 \
#     --output_size 128 \
#     --dropout 0.8
# done

# # tune dropout
# for dropout in 0.3 0.5 0.8; do
# python3 project/train.py \
#     --model_name ACNN \
#     --learning_rate 0.005 \
#     --batch_size 32 \
#     --epochs 100 \
#     --num_filters 100 \
#     --output_size 128 \
#     --dropout $dropout
# done