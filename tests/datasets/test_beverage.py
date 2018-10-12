import unittest
import numpy as np
from libs.datasets.real.beverage import load_data

class TestBeverageDataset(unittest.TestCase):
    def test_import_all_data(self):
        X,label_beverage,label_situation = load_data(ret_beverage_label=True,ret_situation_label=True)
        self.assertTrue(isinstance(X,np.ndarray))
        self.assertEqual(X.ndim,3)
        label_beverage_direct = np.loadtxt('../libs/datasets/real/beverage_data/beverage_label.txt',dtype='str', delimiter=',')
        label_situation_direct = np.loadtxt('../libs/datasets/real/beverage_data/situation_label.txt',dtype='str', delimiter=',')
        self.assertTrue((label_beverage==label_beverage_direct).all())
        self.assertTrue((label_situation==label_situation_direct).all())
        print(label_beverage)
        print(label_situation)

if __name__=='__main__':
    unittest.main()