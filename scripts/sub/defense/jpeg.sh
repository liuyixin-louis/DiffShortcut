
# export INSTANCE_DIR_CHECK_DEFENSED=$INSTANCE_DIR_CHECK-jpeg
source activate $ADB_ENV_NAME
cd $ADB_PROJECT_ROOT/diffshortcut

python3 defenses/jpeg.py \
  --input_dir $1 --output_dir $2 \
  --quality=75