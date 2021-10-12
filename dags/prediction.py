from datetime import timedelta
import os
import yaml

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

import prediction_tasks

with open(
    os.path.join(os.getenv("CONFIG_FOLDER"), "prediction_config.yml")
) as f:
    try:
        prediction_config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": ["airflow@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "prediction",
    default_args=default_args,
    description="Prediction DAG",
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["prediction"],
) as dag:

    start = DummyOperator(task_id="start")
    end = DummyOperator(task_id="end")

    # Try building a DAG that predicts on all three input files under DATA_PREDICTION_INPUT in parallel
    # for example,
    # for batch in prediction_config["pred_set_names"]: