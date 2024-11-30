import pandas as pd
import joblib
import logging
import mlflow
import dagshub
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import TransformedTargetRegressor


# initialize dagshub
import dagshub
dagshub.init(repo_owner='himanshu1703', 
             repo_name='swiggy-delivery-time-prediction', 
             mlflow=True)

# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/himanshu1703/swiggy-delivery-time-prediction.mlflow")

# set mlflow experment name
mlflow.set_experiment("DVC Pipeline")

TARGET = "time_taken"

# create logger
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    
    except FileNotFoundError:
        logger.error("The file to load does not exist")
    
    return df


def make_X_and_y(data:pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def load_model(model_path: Path):
    model = joblib.load(model_path)
    return model


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    # train data load path
    train_data_path = root_path / "data" / "processed" / "train_trans.csv"
    test_data_path = root_path / "data" / "processed" / "test_trans.csv"
    # model path
    model_path = root_path / "models" / "model.joblib"
    
    
    # load the training data
    train_data = load_data(train_data_path)
    logger.info("Train data loaded successfully")
    # load the test data
    test_data = load_data(test_data_path)
    logger.info("Test data loaded successfully")
    
    # split the train and test data
    X_train, y_train = make_X_and_y(train_data,TARGET)
    X_test, y_test = make_X_and_y(test_data,TARGET)
    logger.info("Data split completed")
    
    # load the model
    model = load_model(model_path)
    logger.info("Model Loaded successfully")
    
    
    # get the predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    logger.info("prediction on data complete")
    
    # calculate the train and test mae
    train_mae = mean_absolute_error(y_train,y_train_pred)
    test_mae = mean_absolute_error(y_test,y_test_pred)
    logger.info("error calculated")
    
    # calculate the r2 scores
    train_r2 = r2_score(y_train,y_train_pred)
    test_r2 = r2_score(y_test,y_test_pred)
    logger.info("r2 score calculated")
    
    # calculate cross val scores
    cv_scores = cross_val_score(model,
                                X_train,
                                y_train,
                                cv=5,
                                scoring="neg_mean_absolute_error",
                                n_jobs=-1)
    logger.info("cross validation complete")
    
    # mean cross val score
    mean_cv_score = -(cv_scores.mean())
    
    # log with mlflow
    with mlflow.start_run():
        # set tags
        mlflow.set_tag("model","Food Delivery Time Regressor")

        # log parameters
        mlflow.log_params(model.get_params())

        # log metrics
        mlflow.log_metric("train_mae",train_mae)
        mlflow.log_metric("test_mae",test_mae)
        mlflow.log_metric("train_r2",train_r2)
        mlflow.log_metric("test_r2",test_r2)
        mlflow.log_metric("mean_cv_score",-(cv_scores.mean()))

        # log individual cv scores
        mlflow.log_metrics({f"CV {num}": score for num, score in enumerate(-cv_scores)})
        
        # mlflow dataset input datatype
        train_data_input = mlflow.data.from_pandas(train_data,targets=TARGET)
        test_data_input = mlflow.data.from_pandas(test_data,targets=TARGET)
        
        # log input
        mlflow.log_input(dataset=train_data_input,context="training")
        mlflow.log_input(dataset=test_data_input,context="validation")
        
        # model signature
        model_signature = mlflow.models.infer_signature(model_input=X_train.sample(20,random_state=42),
                                    model_output=model.predict(X_train.sample(20,random_state=42)))
        
        # log the stacking regressor
        mlflow.sklearn.log_model(model,"model",signature=model_signature)
        
        logger.info("Mlflow logging complete and model logged")
