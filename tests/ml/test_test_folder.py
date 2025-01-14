import pickle
import unittest
import torch as pt
from slangmod.ml import TestSequenceFolder


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.seq_len = 512
        self.fold = TestSequenceFolder(self.seq_len)

    def test_has_seq_len(self):
        self.assertTrue(hasattr(self.fold, 'seq_len'))

    def test_seq_len(self):
        self.assertIsInstance(self.fold.seq_len, int)
        self.assertEqual(self.seq_len, self.fold.seq_len)

    def test_non_integer_seq_len_raises(self):
        with self.assertRaises(TypeError):
            _ = TestSequenceFolder('hello world')

    def test_seq_len_too_short_raises(self):
        with self.assertRaises(ValueError):
            _ = TestSequenceFolder(1)

    def test_has_pad_id(self):
        self.assertTrue(hasattr(self.fold, 'pad_id'))

    def test_pad_id(self):
        self.assertIsInstance(self.fold.pad_id, int)
        self.assertEqual(0, self.fold.pad_id)

    def test_has_width(self):
        self.assertTrue(hasattr(self.fold, 'width'))

    def test_width(self):
        self.assertIsInstance(self.fold.width, int)
        self.assertEqual(self.seq_len + 1, self.fold.width)


class TestCustomAttributes(unittest.TestCase):

    def setUp(self):
        self.pad_id = -100
        self.fold = TestSequenceFolder(512, self.pad_id)

    def test_pad_id(self):
        self.assertEqual(self.pad_id, self.fold.pad_id)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.pad_id = -3
        self.fold = TestSequenceFolder(7, self.pad_id)

    def test_callable(self):
        self.assertTrue(callable(self.fold))

    def test_empty_shape(self):
        empty = pt.tensor([])
        actual = self.fold(empty)
        self.assertTupleEqual((1, 8), actual.shape)

    def test_empty_value(self):
        empty = pt.tensor([])
        actual = self.fold(empty)
        expected = self.pad_id * pt.ones(1, 8)
        pt.testing.assert_close(actual, expected)

    def test_one_element_shape(self):
        empty = pt.tensor([1])
        actual = self.fold(empty)
        self.assertTupleEqual((1, 8), actual.shape)

    def test_one_element_value(self):
        empty = pt.tensor([1.0])
        actual = self.fold(empty)
        expected = self.pad_id * pt.ones(1, 8)
        expected[:, 0] = 1.0
        pt.testing.assert_close(actual, expected)

    def test_single_max(self):
        single = pt.arange(1, 9)
        expected = pt.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        actual = self.fold(single)
        pt.testing.assert_close(actual, expected)

    def test_double_min(self):
        double = pt.arange(1, 10)
        expected = pt.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 9] + [self.pad_id] * 6,
        ])
        actual = self.fold(double)
        pt.testing.assert_close(actual, expected)

    def test_double_max(self):
        double = pt.arange(1, 16)
        expected = pt.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 9, 10, 11, 12, 13, 14, 15]
        ])
        actual = self.fold(double)
        pt.testing.assert_close(actual, expected)

    def test_triple_min(self):
        triple = pt.arange(1, 17)
        expected = pt.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [15, 16] + [self.pad_id] * 6,
        ])
        actual = self.fold(triple)
        pt.testing.assert_close(actual, expected)

    def test_triple_max(self):
        triple = pt.arange(1, 23)
        expected = pt.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [15, 16, 17, 18, 19, 20, 21, 22]
        ])
        actual = self.fold(triple)
        pt.testing.assert_close(actual, expected)

    def test_quadruple_min(self):
        quadruple = pt.arange(1, 24)
        expected = pt.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [15, 16, 17, 18, 19, 20, 21, 22],
            [22, 23] + [self.pad_id] * 6,
        ])
        actual = self.fold(quadruple)
        pt.testing.assert_close(actual, expected)

    def test_quadruple_max(self):
        quadruple = pt.arange(1, 30)
        expected = pt.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [15, 16, 17, 18, 19, 20, 21, 22],
            [22, 23, 24, 25, 26, 27, 28, 29]
        ])
        actual = self.fold(quadruple)
        pt.testing.assert_close(actual, expected)

    def test_quintuple_min(self):
        quintuple = pt.arange(1, 31)
        expected = pt.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [15, 16, 17, 18, 19, 20, 21, 22],
            [22, 23, 24, 25, 26, 27, 28, 29],
            [29, 30] + [self.pad_id] * 6,
        ])
        actual = self.fold(quintuple)
        pt.testing.assert_close(actual, expected)

    def test_quintuple_max(self):
        quintuple = pt.arange(1, 37)
        expected = pt.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 9, 10, 11, 12, 13, 14, 15],
            [15, 16, 17, 18, 19, 20, 21, 22],
            [22, 23, 24, 25, 26, 27, 28, 29],
            [29, 30, 31, 32, 33, 34, 35, 36]
        ])
        actual = self.fold(quintuple)
        pt.testing.assert_close(actual, expected)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        fold = TestSequenceFolder(512)
        expected = 'TestSequenceFolder(512, 0)'
        self.assertEqual(expected, repr(fold))

    def test_custom_repr(self):
        fold = TestSequenceFolder(512, -100)
        expected = 'TestSequenceFolder(512, -100)'
        self.assertEqual(expected, repr(fold))

    def test_pickle_works(self):
        fold = TestSequenceFolder(512, -100)
        _ = pickle.loads(pickle.dumps(fold))


if __name__ == '__main__':
    unittest.main()
