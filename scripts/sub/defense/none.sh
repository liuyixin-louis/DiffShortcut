

# export INSTANCE_DIR_CHECK_DEFENSED=$INSTANCE_DIR_CHECK
# make $2 if not exists
if [ -d "$2" ]; then
  echo "Directory $2 exists."
else
    mkdir -p $2 
fi

cp -r $1/* $2