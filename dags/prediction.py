from datetime import timedelta
import os
import yaml

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG
from textwrap import dedent

# Operators; we need this to operate!
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

import prediction_tasks

# Read the "prediction_config.yml" into a dictionary
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

    batch_name = prediction_config["pred_set_names"][0]
    prediction_config["input"] = prediction_config["input_file"].format(batch_name)
    prediction_config["output"] = prediction_config["output_file"].format(batch_name)

    get_input = PythonOperator(
        task_id=f"fetch_input_{batch_name}",
        python_callable=prediction_tasks.get_input,
        op_kwargs=prediction_config,
    )
    predict = PythonOperator(
        task_id=f"predict_{batch_name}",
        python_callable=prediction_tasks.prediction,
        op_kwargs=prediction_config,
    )
    output_result = PythonOperator(
        task_id=f"output_result_{batch_name}",
        python_callable=prediction_tasks.output_result,
        op_kwargs=prediction_config,
    )

    dag.doc_md = __doc__
    dag.doc_md = """
        Prediction DAG
        """

    start >> get_input >> predict >> output_result >> end

    # Now, your task is to try modifying the above DAG that predicts on all three input files under
    # /data/prediction_input in parallel

