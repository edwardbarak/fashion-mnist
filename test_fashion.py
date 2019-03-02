import unittest
import fashion

x_train shape: (60000, 28, 28) y_train shape: (60000,)

class TestDatasetShapes(unittest.TestCase):
    def test_x_train_shape(self):
        self.assertEqual(fashion.x_train.shape == (60000, 28, 28))

    def test_y_train_shape(self):
        self.assertEqual(fashion.y_train.shape == (60000,))
