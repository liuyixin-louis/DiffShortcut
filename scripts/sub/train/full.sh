export train_mode_name=$ft_type
export GEN_IMG_OUTPUT_DIR="$ADB_PROJECT_ROOT/exp_data/train_output/$train_exp_name/$train_hyper/${instance_name}/$train_mode_name"

# this is to indicate that whether we have finished the training before 
training_finish_indicator=$GEN_IMG_OUTPUT_DIR/finished.txt
# echo $INSTANCE_DIR_CHECK_DEFENSED

# skip training if instance data not exist 
if [ ! -d "$INSTANCE_DIR_CHECK_DEFENSED" ]; then
  echo "instance data not exist, skip training"
  exit 1
fi

# if training_finish_indicator
if [ -f "$training_finish_indicator" ]; then
  echo "training finished before, skip training"
else
# echo $GEN_IMG_OUTPUT_DIR

## training config 
# export max_train_steps=1000
if [ -z "$max_train_steps" ]; then
  max_train_steps=1000
fi 

# prior_loss_weight
if [ -z "$prior_loss_weight" ]; then
  prior_loss_weight=1.0
fi

if [ -z "$lr_scheduler" ]; then
 lr_scheduler=constant
fi

if [ -z "$lr" ]; then
  lr=5e-7
fi

if [ -z "$poison_rate" ]; then 
  poison_rate=1.0
fi



cd $ADB_PROJECT_ROOT/diffshortcut
source activate $ADB_ENV_NAME;

command="""python3 train_dreambooth.py --clean_img_dir $CLEAN_INSTANCE_DIR --clean_ref_db $CLEAN_REF  --instance_name $instance_name --dataset_name $dataset_name --class_name '$class_name' \
  --wandb_entity_name $wandb_entity_name \
  --seed $seed \
  --train_text_encoder \
  --exp_name $train_exp_name \
  --gradient_checkpointing \
  --exp_hyper $train_hyper \
  --pretrained_model_name_or_path='$MODEL_PATH'  \
  --instance_data_dir='$INSTANCE_DIR_CHECK_DEFENSED' \
  --class_data_dir='$CLASS_DIR' \
  --output_dir=$GEN_IMG_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=$prior_loss_weight \
  --instance_prompt='${instance_prompt}' \
  --class_prompt='a photo of ${class_name}' \
  --inference_prompts='${eval_prompts}' \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=$lr \
  --lr_scheduler=$lr_scheduler \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=$max_train_steps \
  --center_crop \
  --sample_batch_size=4 \
  --use_8bit_adam \
  --poison_rate $poison_rate \
  --mixed_precision=bf16 --prior_generation_precision=bf16
  """

  if [ $eval_model_name = "SD21base" ] || [ $eval_model_name = "SD21" ]; then
    command="$command --enable_xformers_memory_efficient_attention"
  fi 


  command="$command --inputNegTok $inputNegTok --outputNeg $outputNeg"
  # if save_model
  if [ ! -z "$save_model" ]; then
    command="$command --save_model"
  fi
  # validation_prompt
  if [ ! -z "$validation_prompt" ]; then
    command="$command --validation_prompt='$validation_prompt'"
  fi
  # validation_steps
  if [ ! -z "$validation_steps" ]; then
    command="$command --validation_steps $validation_steps"
  fi

  # if checkpointing_steps
  if [ ! -z "$checkpointing_steps" ]; then
    command="$command --checkpointing_steps  $checkpointing_steps"
  fi

  # num_validation_images
  if [ ! -z "$num_validation_images" ]; then
    command="$command --num_validation_images $num_validation_images"
  fi

  # noise_prompt
  if [ ! -z "$noise_prompt" ]; then
    command="$command --inputNegTok_Str '$noise_prompt'"
  fi

  # es 
  if [ ! -z "$es" ]; then
    if [ "$es" = "True" ]; then
    command="$command --es"
    fi 
  fi

  # es_metric
  if [ ! -z "$es_metric" ]; then
    command="$command --es_metric $es_metric"
  fi

  # es_steps
  if [ ! -z "$es_steps" ]; then
    command="$command --es_steps $es_steps"
  fi

  
echo $command
eval $command 
fi