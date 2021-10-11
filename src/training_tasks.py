import os
import pathlib
import shutil
from hyperopt import Trials, hp, STATUS_OK, tpe, fmin
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import pyarrow as pa
from pyarrow import parquet

# Helper methods


def _1d_nparray_to_parquet(array, path):
    table = pa.Table.from_arrays(
        [array],
        ["0"],
    )
    parquet.write_table(table, path)


def _2d_nparray_to_parquet(array, path):
    table = pa.Table.from_arrays(
        array,
        names=[str(i) for i in range(len(array))],
    )  # give names to each columns
    parquet.write_table(table, path)


def _parquet_to_2d_nparray(path):
    table_from_parquet = parquet.read_table(path)
    matrix_from_parquet = table_from_parquet.to_pandas().T.to_numpy()
    return matrix_from_parquet


def _parquet_to_1d_nparray(path):
    table_from_parquet = parquet.read_table(path)
    matrix_from_parquet = table_from_parquet.to_pandas().T.to_numpy()
    return matrix_from_parquet[0]


def data_extraction(run_id, **context):
    """
    Extract data
    In this example, it will copy data from source folder to intermedia folder
    """
    print(f"Run data_extraction with run_id: {run_id}")
    assert os.environ.get("DATA_TRAINING_FOLDER")
    assert os.environ.get("DATA_INTERMEDIA_FOLDER")
    from_path = pathlib.Path(
        os.environ.get("DATA_TRAINING_FOLDER"), "winequality-red.csv"
    )
    to_path = pathlib.Path(os.environ.get("DATA_INTERMEDIA_FOLDER"), run_id)
    to_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(from_path, to_path)


def data_validation(run_id, input_file_name, **context):
    """Data validation"""
    assert os.environ.get("DATA_INTERMEDIA_FOLDER")
    input_path = pathlib.Path(
        os.environ.get("DATA_INTERMEDIA_FOLDER"),
        run_id,
        input_file_name,
    )
    wine = pd.read_csv(input_path)
    print(wine.head())
    print(wine.info())


def data_preparation(run_id, **context):
    """Data preparation"""

    assert os.environ.get("DATA_INTERMEDIA_FOLDER")
    run_path = pathlib.Path(os.environ.get("DATA_INTERMEDIA_FOLDER"), run_id)
    input_path = pathlib.Path(
        run_path,
        "winequality-red.csv",
    )
    wine = pd.read_csv(input_path)

    # Separate the dataset as response/target variable and feature variables
    X = wine.drop("quality", axis=1)
    y = wine["quality"]

    # Train and test splitting of data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Applying Standard scaling to get optimized result
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    print(f"x train shape: {X_train.shape}")
    print(f"x train head: {X_train[:10]}")
    print(f"x test shape: {X_test.shape}")
    print(f"y train shape: {y_train.shape}")
    print(f"y train head: {y_train[:10]}")
    print(f"y test shape: {y_test.shape}")

    # Save the train and test set to parquet for next task
    _2d_nparray_to_parquet(X_train, pathlib.Path(run_path, "x_train.parquet"))
    _2d_nparray_to_parquet(X_test, pathlib.Path(run_path, "x_test.parquet"))
    _1d_nparray_to_parquet(y_train, pathlib.Path(run_path, "y_train.parquet"))
    _1d_nparray_to_parquet(y_test, pathlib.Path(run_path, "y_test.parquet"))


def model_training(run_id, tuning, **context):
    """Model training"""
    assert os.environ.get("DATA_INTERMEDIA_FOLDER")
    run_path = pathlib.Path(os.environ.get("DATA_INTERMEDIA_FOLDER"), run_id)
    X_train = _parquet_to_2d_nparray(pathlib.Path(run_path, "x_train.parquet"))
    X_test = _parquet_to_2d_nparray(pathlib.Path(run_path, "x_test.parquet"))
    y_train = _parquet_to_1d_nparray(pathlib.Path(run_path, "y_train.parquet"))
    y_test = _parquet_to_1d_nparray(pathlib.Path(run_path, "y_test.parquet"))
    print(f"x train shape: {X_train.shape}")
    print(f"x train head: {X_train[:10]}")
    print(f"x test shape: {X_test.shape}")
    print(f"y train shape: {y_train.shape}")
    print(f"y train head: {y_train[:10]}")
    print(f"y test shape: {y_test.shape}")

    def hyperopt_train_test(params):
        clf = RandomForestClassifier(**params)
        best_score = cross_val_score(clf, X_train, y_train).mean()
        return {"loss": best_score, "params": params, "status": STATUS_OK}

    space4rf = {
        "max_depth": hp.choice(
            "max_depth",
            range(
                tuning["max_depth"]["min_range"],
                tuning["max_depth"]["max_range"],
            ),
        ),
        "max_features": hp.choice(
            "max_features",
            range(
                tuning["max_features"]["min_range"],
                tuning["max_features"]["max_range"],
            ),
        ),
        "n_estimators": hp.choice(
            "n_estimators",
            range(
                tuning["n_estimators"]["min_range"],
                tuning["n_estimators"]["max_range"],
            ),
        ),
        "criterion": hp.choice("criterion", tuning["criterion"]),
    }

    trials = Trials()
    fmin(
        fn=hyperopt_train_test,
        space=space4rf,
        algo=tpe.suggest,
        max_evals=tuning["max_evals"],
        trials=trials,
    )
    print("Hyperparameter tuning complete")
    best_model = trials.results[np.argmin([r["loss"] for r in trials.results])]
    params = best_model["params"]
    print("Best hyperparamaters: ", params)

    rfc = RandomForestClassifier(**params)
    rfc.fit(X_train, y_train)
    # Save model as "random_forest_model.sav"
    joblib.dump(rfc, pathlib.Path(run_path, "random_forest_model.sav"))


def model_evaluation(run_id, **context):
    """Model evaluation"""
    assert os.environ.get("DATA_INTERMEDIA_FOLDER")
    run_path = pathlib.Path(os.environ.get("DATA_INTERMEDIA_FOLDER"), run_id)
    X_train = _parquet_to_2d_nparray(pathlib.Path(run_path, "x_train.parquet"))
    y_train = _parquet_to_1d_nparray(pathlib.Path(run_path, "y_train.parquet"))
    rfc = joblib.load(pathlib.Path(run_path, "random_forest_model.sav"))

    # Some evaluation for random forest model using cross validation.
    rfc_eval = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=10)
    print(f"Random forest model: {rfc_eval.mean()}")


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def model_validation(run_id, model_tracking, **context):
    """Model validation"""
    assert os.environ.get("DATA_INTERMEDIA_FOLDER")
    run_path = pathlib.Path(os.environ.get("DATA_INTERMEDIA_FOLDER"), run_id)
    X_test = _parquet_to_2d_nparray(pathlib.Path(run_path, "x_test.parquet"))
    y_test = _parquet_to_1d_nparray(pathlib.Path(run_path, "y_test.parquet"))

    rfc = joblib.load(pathlib.Path(run_path, "random_forest_model.sav"))

    pred_rfc = rfc.predict(X_test)

    # Print classification report
    print("\n" + classification_report(y_test, pred_rfc))

    with mlflow.start_run(run_name=run_id):
        (rmse, mae, r2) = eval_metrics(y_test, pred_rfc)
        mlflow.log_params(rfc.get_params())
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(
            sk_model=rfc,
            artifact_path=model_tracking["artifact_path"],
            registered_model_name=model_tracking["registered_model_name"],
        )
