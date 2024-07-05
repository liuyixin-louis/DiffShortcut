# export defense_policy="gau-sr"

INPUT_IMG_BEFORE_PURIFY=$INSTANCE_DIR_CHECK
OUTPUT_IMG_AFTER_PURIFY=$INSTANCE_DIR_CHECK-$defense_policy
# REGEN=False
# if [ -z "$REGEN" ]; then
#   REGEN=False
# fi

if [ -z "$REGEN" ]; then
  REGEN=False
fi

if [ "$REGEN" = "True" ]; then
  curr_policy_add=''
  IMG_TMP_DIR=$INPUT_IMG_BEFORE_PURIFY
  echo "the path of input" $IMG_TMP_DIR
  echo "the path of output" $OUTPUT_IMG_AFTER_PURIFY
  defense_policy_arr=(${defense_policy//-/ })
  for policy in "${defense_policy_arr[@]}"
  do 
    if [ -z "$curr_policy_add" ]; then
      curr_policy_add=$policy
    else
      curr_policy_add=$curr_policy_add-$policy
    fi
      IMG_TMP_DIR_NEXT=$INPUT_IMG_BEFORE_PURIFY-$curr_policy_add
      command="bash $ADB_PROJECT_ROOT/scripts/sub/defense/$policy.sh $IMG_TMP_DIR $IMG_TMP_DIR_NEXT"
      echo $command
      eval $command
      return_code=$?
      if [ $return_code -ne 0 ]; then
        echo "Error: defense $policy failed with return code $return_code"
        exit 1
      fi
      IMG_TMP_DIR=$IMG_TMP_DIR_NEXT
  done 
  # assert $IMG_TMP_DIR == $OUTPUT_IMG_AFTER_PURIFY
  if [ "$IMG_TMP_DIR" != "$OUTPUT_IMG_AFTER_PURIFY" ]; then
    echo "Error: $IMG_TMP_DIR != $OUTPUT_IMG_AFTER_PURIFY"
    exit 1
  fi
  export INSTANCE_DIR_CHECK_DEFENSED=$IMG_TMP_DIR
else
    
  if [ -d "$OUTPUT_IMG_AFTER_PURIFY" ]; then
    echo "Directory $OUTPUT_IMG_AFTER_PURIFY exists; skipping defense."

    export INSTANCE_DIR_CHECK_DEFENSED=$OUTPUT_IMG_AFTER_PURIFY
    # exit 0
  else
    # split them with - and process them one by one 
    curr_policy_add=''
    IMG_TMP_DIR=$INPUT_IMG_BEFORE_PURIFY
    echo "the path of input" $IMG_TMP_DIR
    echo "the path of output" $OUTPUT_IMG_AFTER_PURIFY
    defense_policy_arr=(${defense_policy//-/ })
    for policy in "${defense_policy_arr[@]}"
    do 
      if [ -z "$curr_policy_add" ]; then
        curr_policy_add=$policy
      else
        curr_policy_add=$curr_policy_add-$policy
      fi
        IMG_TMP_DIR_NEXT=$INPUT_IMG_BEFORE_PURIFY-$curr_policy_add
        if [ -d "$IMG_TMP_DIR_NEXT" ]; then
          # if the directory content any png or jpg file, then actually skip otherwise generate
          # Check for .png or .jpg files in the specified directory
          if ls "$IMG_TMP_DIR_NEXT"/*.png "$IMG_TMP_DIR_NEXT"/*.jpg 1> /dev/null 2>&1; then
            echo "The directory '$directory' contains PNG or JPG files."
            continue
          else
            command="bash $ADB_PROJECT_ROOT/scripts/sub/defense/$policy.sh $IMG_TMP_DIR $IMG_TMP_DIR_NEXT"
          fi
          echo "Directory $IMG_TMP_DIR_NEXT exists; skipping genearting."
        else
          command="bash $ADB_PROJECT_ROOT/scripts/sub/defense/$policy.sh $IMG_TMP_DIR $IMG_TMP_DIR_NEXT"
        fi 
        echo $command
        eval $command
        return_code=$?
        if [ $return_code -ne 0 ]; then
          echo "Error: defense $policy failed with return code $return_code"
          exit 1
        fi
        IMG_TMP_DIR=$IMG_TMP_DIR_NEXT
    done 
    # assert $IMG_TMP_DIR == $OUTPUT_IMG_AFTER_PURIFY
    if [ "$IMG_TMP_DIR" != "$OUTPUT_IMG_AFTER_PURIFY" ]; then
      echo "Error: $IMG_TMP_DIR != $OUTPUT_IMG_AFTER_PURIFY"
      exit 1
    fi
    export INSTANCE_DIR_CHECK_DEFENSED=$IMG_TMP_DIR
  fi 

fi
