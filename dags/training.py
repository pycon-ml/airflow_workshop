from datetime import timedelta
import os
from textwrap import dedent
import yaml

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

import training_tasks

with open(
    os.path.join(os.getenv("CONFIG_FOLDER"), "training_config.yml")
) as f:
    try:
        training_config = yaml.safe_load(f)
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
    "training",
    default_args=default_args,
    description="Training DAG",
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["training"],
) as dag:

    data_extraction = PythonOperator(
        task_id="data_extraction",
        python_callable=training_tasks.data_extraction,
        op_kwargs=training_config,
    )
    data_validation = PythonOperator(
        task_id="data_validation",
        python_callable=training_tasks.data_validation,
        op_kwargs=training_config,
    )
    data_preparation = PythonOperator(
        task_id="data_preparation",
        python_callable=training_tasks.data_preparation,
        op_kwargs=training_config,
    )
    model_training = PythonOperator(
        task_id="model_training",
        python_callable=training_tasks.model_training,
        op_kwargs=training_config,
    )
    model_evaluation = PythonOperator(
        task_id="model_evaluation",
        python_callable=training_tasks.model_evaluation,
        op_kwargs=training_config,
    )
    model_validation = PythonOperator(
        task_id="model_validation",
        python_callable=training_tasks.model_validation,
        op_kwargs=training_config,
    )

    data_extraction.doc_md = dedent(
        """\
    #### Task Documentation
    This task copies data from source folder to intermedia folder

    """
    )

    data_validation.doc_md = dedent(
        """\
    #### Task Documentation
    This task prints some rows from the input data

    """
    )

    data_preparation.doc_md = dedent(
        """\
    #### Task Documentation
    This task splits the data into train and test and save them as
    parquet files
    """
    )

    dag.doc_md = __doc__
    dag.doc_md = """
    Training DAG
    """  # otherwise, type it like this

    (
        data_extraction
        >> data_validation
        >> data_preparation
        >> model_training
        >> model_evaluation
        >> model_validation
    )
