
# export 2=$INSTANCE_DIR_CHECK-diffpure
# source activate $ADB_ENV_NAME
# cd $ADB_PROJECT_ROOT/diffshortcut

# echo $file_name_this
# if 2 exsit then skip 
if [ -d "$2" ]; then
  echo "Directory $2 exists."
  exit 0
fi
cd $ADB_PROJECT_ROOT/diffshortcut
$GDiff_ENV_PYTHON defenses/diffpure_main.py \
  --input_dir $1 --output_dir $2 \
  --opt_t 300 \
  --class_name $class_name --empty_condition