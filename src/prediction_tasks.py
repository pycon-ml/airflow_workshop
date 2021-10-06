def get_input(*args, **kwargs):
    """Get input
    In this example, it will copy data from source folder to intermedia folder
    """
    print(f'Run data_extraction with run_id: {kwargs["dag_run"].run_id}')


def prediction(*args, **kwargs):
    """Prediction
    In this example, it will use the input data on intermedia folder and run model to predict
    """
    print(f'Run data_extraction with run_id: {kwargs["dag_run"].run_id}')


def output_result(*args, **kwargs):
    """Extract data
    In this example, it will copy predict result to output folder
    """
    print(f'Run data_extraction with run_id: {kwargs["dag_run"].run_id}')
