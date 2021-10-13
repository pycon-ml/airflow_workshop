
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