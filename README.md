# Business Intelligence Project: SF Crime

This project is about the Kaggle challenge: San Francisco Crime Classification, which can be found on that [link](https://www.kaggle.com/c/sf-crime). With the CRISP-DM process the data will be analyzed and crime categories are predicted using four models: K-NN, Decision Tree, Random Forest and XGBoost. The Jupyter notebooks show the results of each model.

This project also includes a python script which can predict custom data with the CLI. To be able to run this CLI tool, you need to train the desired model. For that, see the steps necessary to train and execute a model.

## Steps necessary before starting

### Working with a devcontainer
- Docker needs to be installed and running
- Devcontainer VSCode extension installed
[Don't know how to use devcontainer?](https://microsoft.github.io/vscode-essentials/en/09-dev-containers.html)

- You need a Kaggle key which will be requested in a prompt to download the Kaggle dataset

### Working without a devcontainer
- Download the Kaggle dataset and create a folder "data" in the project root directory
- Paste the test.csv and train.csv inside the created "data" directory
- Install all python requirements with `pip3 install --user -r requirements.txt`


## How the project is structered
The CRISP-DM process stepps are processed in files and directories starting with a number like "01-data-understanding.ipynb". "02-data-preparation.ipynb" will preprocess the data and create new .csv files in "data/tmp" which are necessary to run the models.

## How to train and execute models
Before you can train the models, make sure that the data preparation step has been executed. This creates the preprocessed data in the "data/tmp" directory. After that you can execute the Jupyter notebooks in the "03-modelling" directory.

## How to execute the CLI prediction tool
The CLI prediction tool is implemented in the file "predictor.py". 

Sample execution of this script:

`python ./predicter.py -m 4 -a 600 block of Montgomery ST --lat 37.795116 --long -122.402793`

| Argument           | Description                                                                                       |
|--------------------|---------------------------------------------------------------------------------------------------|
| -m                 | Model which should be executed. 1: Decision Tree, 2: KNN (25-NN), 3: Random Forest, 4: XGBoost    |
| -a                 | Address of the location                                                                           |
| -t                 | Time in format: YYY-MM-DD hh:mm:ss, default is current time                                       |
| --long             | Longditute decimal                                                                                |
| --lat              | Latitude decimal                                                                                  |
| -d                 | District of the crime                                                                             |
| -lm                | Lists all available models                                                                        |

