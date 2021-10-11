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
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}
with DAG(
    "training",
    default_args=default_args,
    description="Training DAG",
    schedule_interval=None,
    start_date=days_ago(2),
    tags=["example", "training"],
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
    You can document your task using the attributes `doc_md` (markdown),
    `doc` (plain text), `doc_rst`, `doc_json`, `doc_yaml` which gets
    rendered in the UI's Task Instance Details page.
    ![img](http://montcs.bloomu.edu/~bobmon/Semesters/2012-01/491/import%20soul.png)

    """
    )

    dag.doc_md = __doc__
    dag.doc_md = """
    This is a documentation placed anywhere
    """  # otherwise, type it like this

    (
        data_extraction
        >> data_validation
        >> data_preparation
        >> model_training
        >> model_evaluation
        >> model_validation
    )
