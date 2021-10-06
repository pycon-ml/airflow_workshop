import collections

PredictionTask = collections.namedtuple(
    "PredictionTask", ["name", "input_file", "output_file"]
)

tasks = [
    PredictionTask("Wine Set One", "wine_set_one.csv", "wine_set_one_quality.csv"),
    PredictionTask("Wine Set Two", "wine_set_two.csv", "wine_set_two_quality.csv"),
    PredictionTask(
        "Wine Set Three", "wine_set_three.csv", "wine_set_three_quality.csv"
    ),
]
