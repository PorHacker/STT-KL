# same setting
python speech_to_text_bpe.py \
    --config-path=/AN202_data12t/03_STT_FOR_SK/03_KIM/13-Kolon-Batch-Pipeline/Finetune/src/03.train/conf  \
    --config-name=conformer_ctc_bpe_unigram_large_26500_epoch16_finetuning_SKShieldus.yaml  \
    model.train_ds.manifest_filepath=/AN202_data12t/03_STT_FOR_SK/03_KIM/13-Kolon-Batch-Pipeline/Finetune/Manifest/filtering_SKShieldus_manifest_train.json  \
    model.validation_ds.manifest_filepath=/AN202_data12t/03_STT_FOR_SK/03_KIM/13-Kolon-Batch-Pipeline/Finetune/Manifest/filtering_SKShieldus_manifest_validation.json  \
    model.tokenizer.dir=/AN202_data12t/03_STT_FOR_SK/03_KIM/13-Kolon-Batch-Pipeline/Finetune/src/03.train/create_tokens/tok_output_unigram/tokenizer_spe_unigram_v2354  \
    model.tokenizer.type=bpe  
    # exp_manager.create_wandb_logger=True  \
    # exp_manager.wandb_logger_kwargs.name="CTC_Large_Finetuning_2354_Epoch16_2" \
    # exp_manager.wandb_logger_kwargs.project="aithe_2024_combined_training_26500_ctc_large"



# same setting
# python speech_to_text_bpe.py \
#     --config-path=/data/20-Mario/SK-STT/src/train/conf  \
#     --config-name=conformer_ctc_bpe_unigram_large_26500_epoch16_finetuning_2.yaml  \
#     model.train_ds.manifest_filepath=/data/20-Mario/SK-STT/dataset/clean_sk_manifest_train.json  \
#     model.validation_ds.manifest_filepath=/data/20-Mario/SK-STT/dataset/clean_sk_manifest_validation.json  \
#     model.tokenizer.dir=/data/20-Mario/SK-STT/src/create_tokens/tok_output_unigram/tokenizer_spe_unigram_v2354  \
#     model.tokenizer.type=bpe  
#     # exp_manager.create_wandb_logger=True  \
#     # exp_manager.wandb_logger_kwargs.name="CTC_Large_Finetuning_2354_Epoch16_2" \
#     # exp_manager.wandb_logger_kwargs.project="aithe_2024_combined_training_26500_ctc_large"


###################################################
# python speech_to_text_ctc_bpe.py \
#     --config-path=/data3/stt_241118/02_finetune/src/train/conf  \
#     --config-name=conformer_ctc_bpe_unigram_large_26500_epoch16_finetuning.yaml  \
#     model.train_ds.manifest_filepath=/data3/stt_241118/02_finetune/manifest/train_kolon_data_manifest_AN208.json  \
#     model.validation_ds.manifest_filepath=/data3/stt_241118/02_finetune/manifest/dev_kolon_data_manifest_AN208.json  \
#     model.tokenizer.dir=/data3/stt_241118/02_finetune/src/create_tokens/tok_output_unigram/tokenizer_spe_unigram_v2354  \
#     model.tokenizer.type=bpe  \
#     exp_manager.create_wandb_logger=True  \
#     exp_manager.wandb_logger_kwargs.name="CTC_Large_Finetuning_2354_Epoch16" \
#     exp_manager.wandb_logger_kwargs.project="aithe_2024_combined_training_26500_ctc_large"


# lr: 1e-4 -> 1e-3
# python speech_to_text_ctc_bpe.py \
#     --config-path=/data3/stt_241118/02_finetune/src/train/conf  \
#     --config-name=conformer_ctc_bpe_unigram_large_26500_epoch16_finetuning.yaml  \
#     model.train_ds.manifest_filepath=/data3/stt_241118/02_finetune/manifest/train_kolon_data_manifest_AN208.json  \
#     model.validation_ds.manifest_filepath=/data3/stt_241118/02_finetune/manifest/dev_kolon_data_manifest_AN208.json  \
#     model.tokenizer.dir=/data3/stt_241118/02_finetune/src/create_tokens/tok_output_unigram/tokenizer_spe_unigram_v2354  \
#     model.tokenizer.type=bpe  \
#     exp_manager.create_wandb_logger=True  \
#     exp_manager.wandb_logger_kwargs.name="CTC_Large_Finetuning_2354_Epoch16_Lr1e3" \
#     exp_manager.wandb_logger_kwargs.project="aithe_2024_combined_training_26500_ctc_large"
