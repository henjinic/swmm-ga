import os
import pathlib
from datetime import datetime


class GALogger:

    def __init__(self, base_dir, model_name):
        """model_name: "now" -> `datetime.now().strftime("%Y-%m-%d_%H-%M-%S")`"""
        if model_name == "now":
            model_name = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        self._model_dir = pathlib.Path(base_dir, model_name)
        if not self._model_dir.exists():
            self._model_dir.mkdir(parents=True)

        self._costs_path = (self._model_dir / "costs.csv")
        with self._costs_path.open("w") as f:
            f.write("generation,rank,costs\n")

    def log(self, generation, ranking, gene, costs):
        dir_path = (self._model_dir / str(generation))

        if not dir_path.exists():
            dir_path.mkdir()

        with self._costs_path.open("a") as f:
            f.write(",".join(map(str, [generation, ranking] + costs)))
            f.write("\n")

        with (dir_path / str(ranking)).with_suffix(".csv").open("w") as f:
            for line in gene:
                f.write(",".join(map(str, line)))
                f.write("\n")


class GALogger27:

    def __init__(self, base_dir, model_name):
        """model_name: "now" -> `datetime.now().strftime("%Y-%m-%d_%H-%M-%S")`"""
        if model_name == "now":
            model_name = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        self._model_dir = os.path.join(base_dir, model_name)
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)

        self._costs_path = os.path.join(self._model_dir, "costs.csv")
        with open(self._costs_path, "w") as f:
            f.write("generation,rank,costs\n")

    def log(self, generation, ranking, gene, costs):
        dir_path = os.path.join(self._model_dir, generation)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(self._costs_path, "a") as f:
            f.write(",".join(map(str, [generation, ranking] + costs)))
            f.write("\n")

        with open(os.path.join(dir_path, str(ranking) + ".csv"), "w") as f:
            for line in gene:
                f.write(",".join(map(str, line)))
                f.write("\n")


def main():
    logger = GALogger("D:/_swwm_results2", "now")

    gene1 = [[1, 2],[3, 4]]
    gene2 = [[5, 6],[7, 8]]

    logger.log(1, 1, gene1, [2, 2, 4])
    logger.log(1, 2, gene2, [3, 3, 2])
    logger.log(2, 1, gene1, [2, 2, 1])
    logger.log(2, 2, gene2, [0, 2, 4])


if __name__ == "__main__":
    main()
