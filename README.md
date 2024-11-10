Please note that the template was changed a little bit to simplify the code structure. The original one was too advanced for this project

# Analysis of Electrophysiological signals for diagnosing eye diseases using Machine Learning

## Motivation
This repository provides the code for the analysis of Electroretinogram signals. We are given the dataset of patients belonging to two categories: healthy and non healthy. The goal is to learn a robust classification model for this dataset. We have tried to feature extraction methods namely shapelets and Short Time fourier Transform (STFT) which improved the results than using only the traditional features. For more details please read the report.

## Installation
You don't need any installation assuming you have python virtual environment or Anaconda.

### Clone
Clone this repo to your local machine using `git clone https://mad-srv.informatik.uni-erlangen.de/MadLab/industry-4.0/seminar-i4.0/ss23/diagnosing-eye-diseases.git `

### Setup
Create the anaconda environment with python>=3.9
```
conda create --name erg python=3.9
```
Activate the environment and install the packages using `pip install -r requirements.txt`. If there are issues with the environment setting up or any missing environment, please feel free to open an issue.


## Run
Please make sure to provide the correct paths and hyperparameters in config.yml file. After that running `python main.py` will show the classification scores of four features sets trained with Random Forests: using traditional features only, including shapelet features, including STFT features and finally combined features.

I must mention that restructuring the code and rerunning give different results every time even with the same test set for each model. But the efficacy of the new features is maintained.

Project Organization
------------

    ├── LICENSE
    ├── README.md      
    ├── dataset
    │   ├── erg_data.xlsx  <- ERG dataset
    │
    ├── notebooks          <- Jupyter notebooks including the shapelets analysis and STFT analysis
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>