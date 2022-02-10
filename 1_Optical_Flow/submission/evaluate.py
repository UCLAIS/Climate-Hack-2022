import numpy as np

from climatehack import BaseEvaluator
from model import Model


class Evaluator(BaseEvaluator):

    def predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        assert data.shape == (12, 128, 128)
        
        model = Model(data)
        
        prediction = model.generate()

        assert prediction.shape == (24, 64, 64)

        return prediction


def main():
    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()