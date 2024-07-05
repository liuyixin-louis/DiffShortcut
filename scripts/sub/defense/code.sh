
# export INSTANCE_DIR_CHECK_DEFENSED=$1-code
# source activate $ADB_ENV_NAME
# cd $ADB_PROJECT_ROOT/diffshortcut
#!/bin/bash
# Path to the Python environment and script
PYTHON_ENV="$CodeFormer_PYTHON_ENV"
SCRIPT_PATH="$ADB_PROJECT_ROOT/diffshortcut/defenses/CodeFormer/inference_codeformer.py"

if [ -d "$2" ]; then
  echo "Directory $2 exists."
  exit 0
fi
# Execute the Python script
cd "$(dirname "$SCRIPT_PATH")" || exit

if [ "$class_name" = "person" ]; then
    $PYTHON_ENV "$SCRIPT_PATH" -w 0.5 --input_path "$1" -o "$2" 
    cp $2/final_results/* $2 

else 
    mkdir -p $2
    cp $1/* $2
fi 


echo "CodeFormer processing complete. Check the output in $2."
