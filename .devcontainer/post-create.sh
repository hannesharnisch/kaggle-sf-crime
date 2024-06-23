pip install --upgrade pip
pip install --user -r /workspaces/sf-crime/requirements.txt
download_dataset() {
  if [ ! "$(ls -A data/zips)" ]; then
    echo "Dataset not found."
    mkdir -p data/zips
    
    while true; do
      # Check if Kaggle API key is set in the environment
      if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
        echo "Kaggle API key not found."
        read -p "Enter your Kaggle username: " KAGGLE_USERNAME
        read -p "Enter your Kaggle API key: " KAGGLE_KEY
        echo
      fi
      
      # Export the variables for the current session
      export KAGGLE_USERNAME=$KAGGLE_USERNAME
      export KAGGLE_KEY=$KAGGLE_KEY
      
      # Attempt to download the dataset
      if kaggle competitions download -c sf-crime -p data/zips; then
        echo "Dataset downloaded successfully."

        # Unzip the dataset
        echo "Unzipping dataset..."
        unzip data/zips/sf-crime.zip -d data
        for zip in data/*.zip; do
          unzip "$zip" -d data
          rm "$zip" # Remove the zip file after extraction
        done
        echo "Dataset unzipped."
        echo "Dataset unzipped."
        
        break # Exit the loop if download is successful
      else
        echo "Download failed. Please check your Kaggle API credentials and try again."
        # Clear the variables so the user can re-enter them
        unset KAGGLE_USERNAME
        unset KAGGLE_KEY
      fi
    done
  else
    echo "Dataset already exists."
  fi
}

if [ ! "$(ls -A data)" ]; then
  echo "Dataset not found."
  mkdir data
  download_dataset
else
  echo "Dataset already exists."
fi