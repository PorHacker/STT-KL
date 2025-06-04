python /data2/mario/sk_shielders/03.train/create_tokens/scripts/process_asr_text_tokenizer.py \
   --manifest=/nas2/voice/data/00-MetaM-Project-Backup/01-Wisely/00-Data/for_pre_finetune_eval/all_data_manifest_final-0602.json \
   --data_root=/data2/mario/sk_shielders/03.train/create_tokens/tok_output_unigram \
   --tokenizer=spe \
   --spe_type=unigram \
   --no_lower_case \
   --log \
   --vocab_size=2354



