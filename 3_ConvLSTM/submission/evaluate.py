import numpy as np
import tensorflow as tf

from climatehack import BaseEvaluator


class Evaluator(BaseEvaluator):
    def setup(self):
        """Sets up anything required for evaluation.

        In this case, it loads the trained model (in evaluation mode)."""

        
        self.model = tf.keras.models.load_model('saved_model/my_model')

    def predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Makes a prediction for the next two hours of satellite imagery.

        Args:
            coordinates (np.ndarray): the OSGB x and y coordinates (2, 128, 128)
            data (np.ndarray): an array of 12 128*128 satellite images (12, 128, 128)

        Returns:
            np.ndarray: an array of 24 64*64 satellite image predictions (24, 64, 64)
        """

        assert coordinates.shape == (2, 128, 128)
        assert data.shape == (12, 128, 128)
        
        # (12, 128, 128, 1)
        features = np.expand_dims(data, axis = -1)
        
        # (1, 64, 64, 1)
        last_image = np.expand_dims(tf.image.resize(features[-1], (64, 64)), axis = 0)
        
        # (1, 1, 64, 64, 1)
        last_image = np.expand_dims(last_image, axis = 0) / 1023
        
        prediction = []
        for i in range(24):
            prediction.append(self.model.predict(last_image))
            last_image = prediction[-1] / 1023
            
        prediction = np.array(prediction)
        prediction = np.squeeze(np.squeeze(prediction, axis = 1), axis = 1)
        prediction = np.squeeze(prediction, axis = -1)

        assert prediction.shape == (24, 64, 64)

        return prediction


def main():
    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
