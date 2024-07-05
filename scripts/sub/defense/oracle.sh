
if [ -d "$2" ]; then
  echo "Directory $2 exists."
else
    mkdir -p $2 
fi

cp -r $CLEAN_IMG_DIR/* $2