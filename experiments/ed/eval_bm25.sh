export _TYPER_STANDARD_TRACEBACK=1

script_name="$0"
script_name_no_ext="${script_name%.*}"

timestamp=$(date +'%m-%d-%Y')

variables_path="./data/gsim/survey_items.jsonl"
data_dir="./data/sild"
index_dir=".results/qdrant_indices/"
metadata_values="title,variable_label,question_text,sub_question,item_category,topic,item_categories,answer_categories"

output_dir="./results/ed/$script_name_no_ext"

model="bm25"

# eval on full test dataset & filter & subset
python $ed_script \
    --variables-path=$variables_path \
    --data-dir=$data_dir \
    --index-dir=$index_dir \
    --output-dir=$output_dir"_filter_subset" \
    --do-sparse \
    --no-do-dense \
    --no-do-join \
    --no-do-rerank \
    --do-subset-datastore \
    --doc-store-type="inmemory" \
    --metadata-values=$metadata_values \
    --model-name=$model \
    --do-filter \
    --do-subset-datastore \
    --no-eval-on-gold \
    --no-use-validation-data  # evaluate on test data

# no subset & no filter
python $ed_script \
    --variables-path=$variables_path \
    --data-dir=$data_dir \
    --index-dir=$index_dir \
    --output-dir=$output_dir"_no-subset_no-filter" \
    --do-sparse \
    --no-do-dense \
    --no-do-join \
    --no-do-rerank \
    --do-subset-datastore \
    --doc-store-type="inmemory" \
    --metadata-values=$metadata_values \
    --model-name=$model \
    --no-do-filter \
    --no-do-subset-datastore \
    --no-use-validation-data  # evaluate on test data