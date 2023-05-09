language_pairs="eng_Latn-deu_Latn,eng_Latn-nld_Latn,eng_Latn-spa_Latn,eng_Latn-zho_Hans,eng_Latn-arb_Arab,deu_Latn-eng_Latn,nld_Latn-eng_Latn,spa_Latn-eng_Latn,zho_Hans-eng_Latn,arb_Arab-eng_Latn"

python run_translation_multilingual.py \
    --model_name_or_path  facebook/nllb-200-distilled-1.3B \
    --do_train \
    --learning_rate 0.00005 \
    --max_steps 1000 \
    --warmup_steps 100 \
    --save_steps 100 \
    --source_lang eng_Latn \
    --target_lang zho_Hans \
    --output_dir output/nllb-200-distilled-1.3B-easyproject \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=1 \
    --overwrite_output_dir \
    --predict_with_generate \
    --remove_unused_columns False \
    --freeze_named_params layers.23 \
    --train_file  mt_train_data/{}/train.json \
    --language_pairs $language_pairs \