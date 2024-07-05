export train_mode_name=$ft_type
export GEN_IMG_OUTPUT_DIR="$ADB_PROJECT_ROOT/exp_data/train_output/$train_exp_name/$train_hyper/${instance_name}/$train_mode_name"

# this is to indicate that whether we have finished the training before 
training_finish_indicator=$GEN_IMG_OUTPUT_DIR/finished.txt
echo $INSTANCE_DIR_CHECK_DEFENSED

# skip training if instance data not exist 
if [ ! -d "$INSTANCE_DIR_CHECK_DEFENSED" ]; then
  echo "instance data not exist, skip training"
  exit 1
fi

# if training_finish_indicator
if [ -f "$training_finish_indicator" ]; then
  echo "training finished before, skip training"
  # exit 
# else
fi


