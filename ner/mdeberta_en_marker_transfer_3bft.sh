#!/bin/bash -l
#SBATCH -J masaka
#SBATCH --output=log/masaka%j-%a.out
#SBATCH --error=log/masaka%j-%a.err
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --partition=overcap
#SBATCH --constraint="a40"
#SBATCH --account=overcap
#SBATCH --time=48:00:00	
#SBATCH -a 1-1000
#SBATCH --requeue

sid=$SLURM_ARRAY_TASK_ID
TGTLANG=$(sed -n "${sid}p" masaka_lang_id.txt)
SRCLANG=en
type=nllb_3B_ft_marker
for j in 1 2 3 4 5
do
	export MAX_LENGTH=164
	export BERT_MODEL=microsoft/mdeberta-v3-base
	export OUTPUT_DIR="save_ckpts_conll/${type}_${SRCLANG}_${TGTLANG}_mdeberta_$j"
	export TEXT_RESULT=test_result_${SRCLANG}_$j.txt
	export TEXT_PREDICTION=test_predictions_${SRCLANG}_$j.txt
	export BATCH_SIZE=32
	export NUM_EPOCHS=5
	export SAVE_STEPS=10000
	export SEED=$j
    export data_dir=output_nllb_3Bft_conll/en-${TGTLANG}/

	if test -d "$OUTPUT_DIR/pytorch_model.bin"; then
		echo "exists"
	else
		CUDA_VISIBLE_DEVICES=0 python3 train_ner.py --data_dir $data_dir \
		--model_type xlmroberta \
		--model_name_or_path $BERT_MODEL \
		--output_dir $OUTPUT_DIR \
		--test_result_file $TEXT_RESULT \
		--test_prediction_file $TEXT_PREDICTION \
		--max_seq_length  $MAX_LENGTH \
		--num_train_epochs $NUM_EPOCHS \
		--per_gpu_train_batch_size $BATCH_SIZE \
		--per_gpu_eval_batch_size $BATCH_SIZE \
		--save_steps $SAVE_STEPS \
		--seed $SEED \
		--do_train \
		--overwrite_output_dir
	fi

    export TEXT_RESULT=test_result_${TGTLANG}_$j.txt
    export TEXT_PREDICTION=test_predictions_${TGTLANG}_$j.txt
    export TEXT_RESULT_DEV=dev_result_${TGTLANG}_$j.txt
    export TEXT_PREDICTION_DEV=dev_predictions_${TGTLANG}_$j.txt

	if test -d "$OUTPUT_DIR/$TEXT_RESULT"; then
		echo "exists"
	else
		CUDA_VISIBLE_DEVICES=0 python3 train_ner.py --data_dir data_masakhaner/${TGTLANG}/ \
		--model_type xlmroberta \
		--model_name_or_path $BERT_MODEL \
		--output_dir $OUTPUT_DIR \
		--test_result_file $TEXT_RESULT \
		--test_prediction_file $TEXT_PREDICTION \
		--dev_result_file $TEXT_RESULT_DEV \
		--dev_prediction_file $TEXT_PREDICTION_DEV \
		--max_seq_length  $MAX_LENGTH \
		--num_train_epochs $NUM_EPOCHS \
		--per_gpu_train_batch_size $BATCH_SIZE \
		--per_gpu_eval_batch_size $BATCH_SIZE \
		--save_steps $SAVE_STEPS \
		--seed $SEED \
		--do_eval \
		--do_predict \
		--overwrite_output_dir
	fi
done
