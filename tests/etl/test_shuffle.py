import pickle
import random
import unittest
from unittest.mock import patch
from slangmod.etl import Shuffle


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.shuffle = Shuffle()

    def test_has_active(self):
        self.assertTrue(hasattr(self.shuffle, 'active'))

    def test_active(self):
        self.assertIsInstance(self.shuffle.active, bool)
        self.assertTrue(self.shuffle.active)


class TestCustomAttributes(unittest.TestCase):

    def setUp(self):
        self.shuffle = Shuffle(False)

    def test_active(self):
        self.assertIsInstance(self.shuffle.active, bool)
        self.assertFalse(self.shuffle.active)


class TestUsage(unittest.TestCase):

    @patch('random.shuffle')
    def test_active(self, mock):
        shuffle = Shuffle()
        inp = [1, 2, 3]
        _ = shuffle(inp)
        mock.assert_called_once_with(inp)

    @patch('random.shuffle')
    def test_inactive(self, mock):
        shuffle = Shuffle(False)
        inp = [1, 2, 3]
        _ = shuffle(inp)
        mock.assert_not_called()

    def test_inplace(self):
        shuffle = Shuffle()
        inp = [1, 2, 3]
        out = shuffle(inp)
        self.assertIs(inp, out)

    def test_shuffling(self):
        shuffle = Shuffle[list[int]]()
        inp = [1, 2, 3]
        random.seed(123456)
        out = shuffle(inp)
        self.assertListEqual([3, 1, 2], out)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        shuffle = Shuffle()
        expected = "Shuffle(True)"
        self.assertEqual(expected, repr(shuffle))

    def test_custom_repr(self):
        shuffle = Shuffle(False)
        expected = "Shuffle(False)"
        self.assertEqual(expected, repr(shuffle))

    def test_pickle_works(self):
        shuffle = Shuffle()
        _ = pickle.loads(pickle.dumps(shuffle))


if __name__ == '__main__':
    unittest.main()
