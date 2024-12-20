timestamp=$(date +'%m-%d-%Y_%H-%M')

root_dir="./"
log_dir=$root_dir

models="intfloat/multilingual-e5-base"
epochs=50
seeds="42 1234 1337"

samples="200 2000 20000 200000"
train_path=$root_dir"data/gsim/pairs_train.jsonl"
val_path=$root_dir"data/gsim/pairs_val.jsonl"
for sample in $samples; do
    for seed in $seeds; do
        output_dir=$root_dir"results/filtered/"$timestamp"/sample="$sample"/seed="$seed
        for model in $models; do
            python sosci_simlearn/train.py --train-path=$train_path --val-path=$val_path --output-dir=$output_dir --max-epochs=$epochs --model-name=$model --sample-n=$sample --seed=$seed --log-dir=$log_dir
        done
    done
done

samples="200 2000 20000 200000 400000"
train_path=$root_dir"data/llm-gen/pairs_train.jsonl"
val_path=$root_dir"data/llm-gen/pairs_val.jsonl"
for sample in $samples; do
    for seed in $seeds; do
        output_dir=$root_dir"results/filtered_gen/"$timestamp"/sample="$sample"/seed="$seed
        for model in $models; do
            python sosci_simlearn/train.py --train-path=$train_path --val-path=$val_path --output-dir=$output_dir --max-epochs=$epochs --model-name=$model --sample-n=$sample --seed=$seed --log-dir=$log_dir
        done
    done
done