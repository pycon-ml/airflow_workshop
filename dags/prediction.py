from datetime import timedelta
from textwrap import dedent

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

import prediction_tasks
import prediction_config

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
    tags=["example", "prediction"],
) as dag:

    start = DummyOperator(task_id="start")
    end = DummyOperator(task_id="end")

    for task in prediction_config.tasks:
        get_input = PythonOperator(
            task_id="fetch_input_" + "_".join(task.name.split()).lower(),
            python_callable=prediction_tasks.get_input,
            op_kwargs={"input": task.input_file, "output": task.output_file},
        )
        predict = PythonOperator(
            task_id="predict_" + "_".join(task.name.split()).lower(),
            python_callable=prediction_tasks.prediction,
            op_kwargs={"input": task.input_file, "output": task.output_file},
        )
        output_result = PythonOperator(
            task_id="output_result_" + "_".join(task.name.split()).lower(),
            python_callable=prediction_tasks.output_result,
            op_kwargs={"input": task.input_file, "output": task.output_file},
        )
        start >> get_input >> predict >> output_result >> end

