import os
import pathlib
import shutil
import mlflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_input(run_id, input, **context):
    """Get input
    In this example, it will copy data from source folder to intermedia folder
    """
    print(f"Run get_input with run_id: {run_id}")
    assert os.environ.get("DATA_PREDICTION_INPUT")
    assert os.environ.get("DATA_INTERMEDIA_FOLDER")
    from_path = pathlib.Path(os.environ.get("DATA_PREDICTION_INPUT"), input)
    to_path = pathlib.Path(os.environ.get("DATA_INTERMEDIA_FOLDER"), run_id)
    to_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(from_path, to_path)


def prediction(run_id, input, output, **context):
    """Prediction
    In this example, it will use the input data on intermedia folder and run
    model to predict
    """
    print(f"Run prediction with run_id: {run_id}")
    assert os.environ.get("DATA_INTERMEDIA_FOLDER")
    input_path = pathlib.Path(
        os.environ.get("DATA_INTERMEDIA_FOLDER"),
        run_id,
        input,
    )
    wine = pd.read_csv(input_path)
    print(f"Input data: {wine}")
    # Applying Standard scaling
    sc = StandardScaler()
    wine = sc.fit_transform(wine)
    print(f"Input data after scale: {wine}")
    model = mlflow.sklearn.load_model("models:/ElasticnetWineModel/Production")
    predict_result = model.predict(wine)
    print(f"Predict result type: {type(predict_result)}")
    print(f"Predict result: {predict_result}")
    output_file = pathlib.Path(
        os.environ.get("DATA_INTERMEDIA_FOLDER"),
        run_id,
        output,
    )
    np.savetxt(output_file, predict_result, delimiter=",")


def output_result(run_id, output, **context):
    """Extract data
    In this example, it will copy predict result to output folder
    """
    print(f"Run output_result with run_id: {run_id}")
    assert os.environ.get("DATA_PREDICTION_OUTPUT")
    assert os.environ.get("DATA_INTERMEDIA_FOLDER")
    from_path = pathlib.Path(
        os.environ.get("DATA_INTERMEDIA_FOLDER"),
        run_id,
        output,
    )
    to_path = pathlib.Path(
        os.environ.get("DATA_PREDICTION_OUTPUT"),
    )
    to_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(from_path, to_path)
