from box import Box
import yaml
from src.data.make_dataset import prepare_data, get_Xy
from src.models.train_model import (
    train_traditional,
    train_shapelet,
    train_stft,
    train_combined,
)

if __name__ == "__main__":
    conf_path = "./config.yml"
    # to make the config file in dot notation
    conf = Box.from_yaml(filename=conf_path, Loader=yaml.FullLoader)

    df = prepare_data(conf.data.path)
    # divide df into train and test

    train_signals, test_signals, Xtrain, Xtest, ytrain, ytest = get_Xy(
        df, conf.data.train_columns, conf.data.target_column, conf.data.test_ratio
    )
    ##### traditional features training ########
    train_traditional(Xtrain, Xtest, ytrain, ytest)

    #### shapelet features training ##########
    print("---------- training shapelets only ---------")
    train_shapelet(
        Xtrain,
        Xtest,
        ytrain,
        ytest,
        train_signals,
        test_signals,
        conf.experiments.num_shapelets[1],
    )
    print("--------------training STFT only ----------------------")
    train_stft(
        Xtrain,
        Xtest,
        ytrain,
        ytest,
        train_signals,
        test_signals,
    )
    print("------------------Training Combined ---------------------")
    train_combined(
        Xtrain,
        Xtest,
        ytrain,
        ytest,
        train_signals,
        test_signals,
        conf.experiments.num_shapelets[1],
    )
