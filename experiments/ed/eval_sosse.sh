export _TYPER_STANDARD_TRACEBACK=1

script_name="$0"
script_name_no_ext="${script_name%.*}"

timestamp=$(date +'%m-%d-%Y')

variables_path="./data/gsim/survey_items.jsonl"
data_dir="./data/sild"
index_dir=".results/qdrant_indices/"
metadata_values="title,variable_label,question_text,sub_question,item_category,topic,item_categories,answer_categories"

output_dir="./results/ed/$script_name_no_ext/$timestamp"

model_dir=""

python $ed_script \
  --variables-path=$variables_path \
  --data-dir=$data_dir \
  --index-dir=$index_dir \
  --output-dir=$output_dir \
  --no-do-sparse \
  --no-do-join \
  --no-do-rerank \
  --do-subset-datastore \
  --metadata-values=$metadata_values \
  --model-name=$model_dir \
  --use-prefix \
  --doc-store-type="inmemory" \
  --model-format="quaterion" \
  --no-use-validation-data