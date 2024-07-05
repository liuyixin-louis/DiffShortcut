
# export INSTANCE_DIR_CHECK_DEFENSED=$INSTANCE_DIR_CHECK-jpeg
# source activate $ADB_ENV_NAME
# cd $ADB_PROJECT_ROOT/diffshortcut
cd $ADB_PROJECT_ROOT/diffshortcut
file_name_this=$(basename "${BASH_SOURCE[0]}")
# remove the sh postfix
file_name_this="${file_name_this%.*}"

# echo $file_name_this
python3 defenses/$file_name_this.py \
  --input_dir $1 --output_dir $2 \
  --class_name $class_name