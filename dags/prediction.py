from datetime import timedelta
import os
from textwrap import dedent

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

    for task in prediction_config["pred_set_names"]:
        prediction_config["input"] = prediction_config["input_file"].format(
            task
        )
        prediction_config["output"] = prediction_config["output_file"].format(
            task
        )

        get_input = PythonOperator(
            task_id="fetch_input_" + task,
            python_callable=prediction_tasks.get_input,
            op_kwargs=prediction_config,
        )
        predict = PythonOperator(
            task_id="predict_" + task,
            python_callable=prediction_tasks.prediction,
            op_kwargs=prediction_config,
        )
        output_result = PythonOperator(
            task_id="output_result_" + task,
            python_callable=prediction_tasks.output_result,
            op_kwargs=prediction_config,
        )
        get_input.doc_md = dedent(
            """\
        #### Task Documentation
        This task copies input data into intermediate folder
        """
        )

        predict.doc_md = dedent(
            """
        #### Task Documentation
        This task loads model and predicts on the given input data
        """
        )

        output_result.doc_md = dedent(
            """
        #### Task Documentation
        This task copies prediction output from intermediate to output folder
        """
        )

        dag.doc_md = __doc__
        dag.doc_md = """
            Prediction DAG
            """

        start >> get_input >> predict >> output_result >> end
