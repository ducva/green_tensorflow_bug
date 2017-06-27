import unittest
import numpy as np
from global_service import predictor


class Test(unittest.TestCase):

    def test_case(self):
        test_data = np.array([
            [0,0.5,0,0,0,0,0,0,2,0,0,0,0]
        ])
        labels=predictor.predict(test_data)
        expected = [0]
        self.assertListEqual(labels, expected)

if __name__ == "__main__":
    unittest.main()