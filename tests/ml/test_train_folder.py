import pickle
import unittest
import torch as pt
from slangmod.ml import TrainSequenceFolder, ValidationErrors


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.seq_len = 512
        self.fold = TrainSequenceFolder(self.seq_len)

    def test_has_seq_len(self):
        self.assertTrue(hasattr(self.fold, 'seq_len'))

    def test_seq_len(self):
        self.assertIsInstance(self.fold.seq_len, int)
        self.assertEqual(self.seq_len, self.fold.seq_len)

    def test_non_integer_seq_len_raises(self):
        with self.assertRaises(ValidationErrors):
            _ = TrainSequenceFolder('hello world')

    def test_seq_len_too_short_raises(self):
        with self.assertRaises(ValidationErrors):
            _ = TrainSequenceFolder(1)

    def test_has_pad_id(self):
        self.assertTrue(hasattr(self.fold, 'pad_id'))

    def test_pad_id(self):
        self.assertIsInstance(self.fold.pad_id, int)
        self.assertEqual(0, self.fold.pad_id)

    def test_has_overlap(self):
        self.assertTrue(hasattr(self.fold, 'overlap'))

    def test_overlap(self):
        self.assertIsInstance(self.fold.overlap, int)
        self.assertEqual(0, self.fold.overlap)

    def test_has_jitter(self):
        self.assertTrue(hasattr(self.fold, 'jitter'))

    def test_jitter(self):
        self.assertIsInstance(self.fold.jitter, int)
        self.assertEqual(1, self.fold.jitter)

    def test_has_width(self):
        self.assertTrue(hasattr(self.fold, 'width'))

    def test_width(self):
        self.assertIsInstance(self.fold.width, int)
        self.assertEqual(self.seq_len + 1, self.fold.width)

    def test_has_stride(self):
        self.assertTrue(hasattr(self.fold, 'stride'))

    def test_stride(self):
        self.assertIsInstance(self.fold.stride, int)
        self.assertEqual(self.seq_len, self.fold.stride)

    def test_callable(self):
        self.assertTrue(callable(self.fold))


class TestCustomAttributes(unittest.TestCase):

    def setUp(self):
        self.seq_len = 512
        self.pad_id = -100
        self.int_overlap = 128
        self.float_overlap = 0.25
        self.jitter = 32
        self.fold = TrainSequenceFolder(self.seq_len)

    def test_pad_id(self):
        fold = TrainSequenceFolder(self.seq_len, self.pad_id)
        self.assertEqual(self.pad_id, fold.pad_id)

    def test_raises_non_numeric_overlap(self):
        with self.assertRaises(ValidationErrors):
            _ = TrainSequenceFolder(self.seq_len, self.pad_id, 'overlap')

    def test_raises_negative_float_overlap(self):
        with self.assertRaises(ValidationErrors):
            _ = TrainSequenceFolder(self.seq_len, self.pad_id, -0.1)

    def test_raises_negative_int_overlap(self):
        with self.assertRaises(ValidationErrors):
            _ = TrainSequenceFolder(self.seq_len, self.pad_id, -3)

    def test_raises_int_overlap_seq_len(self):
        with self.assertRaises(ValidationErrors):
            _ = TrainSequenceFolder(self.seq_len, self.pad_id, self.seq_len)

    def test_int_overlap(self):
        fold = TrainSequenceFolder(self.seq_len, self.pad_id, self.int_overlap)
        self.assertIsInstance(fold.overlap, int)
        self.assertEqual(fold.overlap, 128)
        self.assertEqual(fold.stride, 384)

    def test_float_overlap(self):
        fold = TrainSequenceFolder(
            self.seq_len,
            self.pad_id,
            self.float_overlap
        )
        self.assertIsInstance(fold.overlap, int)
        self.assertEqual(fold.overlap, 128)
        self.assertEqual(fold.stride, 384)

    def test_overlap_float_one(self):
        fold = TrainSequenceFolder(self.seq_len, self.pad_id, 1.0)
        self.assertEqual(1, fold.overlap)
        self.assertEqual(self.seq_len, self.fold.stride)

    def test_overlap_int_one(self):
        fold = TrainSequenceFolder(self.seq_len, self.pad_id, 1)
        self.assertEqual(1, fold.overlap)
        self.assertEqual(self.seq_len, self.fold.stride)

    def test_raises_jitter_not_int(self):
        with self.assertRaises(ValidationErrors):
            _ = TrainSequenceFolder(self.seq_len, jitter='jitter')

    def test_raises_jitter_zero(self):
        with self.assertRaises(ValidationErrors):
            _ = TrainSequenceFolder(self.seq_len, jitter=0)

    def test_raises_jitter_negative(self):
        with self.assertRaises(ValidationErrors):
            _ = TrainSequenceFolder(self.seq_len, jitter=-123)

    def test_raises_jitter_too_large(self):
        with self.assertRaises(ValidationErrors):
            _ = TrainSequenceFolder(
                self.seq_len,
                self.pad_id,
                self.int_overlap,
                jitter=385
            )

    def test_jitter(self):
        fold = TrainSequenceFolder(
            self.seq_len,
            self.pad_id,
            self.int_overlap,
            self.jitter
        )
        self.assertEqual(self.jitter, fold.jitter)

    def test_jitter_minimum(self):
        fold = TrainSequenceFolder(
            self.seq_len,
            self.pad_id,
            self.int_overlap,
            jitter=1
        )
        self.assertEqual(1, fold.jitter)

    def test_jitter_maximum(self):
        fold = TrainSequenceFolder(
            self.seq_len,
            self.pad_id,
            self.int_overlap,
            jitter=384
        )
        self.assertEqual(384, fold.jitter)

    def test_width(self):
        fold = TrainSequenceFolder(
            self.seq_len,
            self.pad_id,
            self.int_overlap,
            self.jitter
        )
        self.assertEqual(self.seq_len + self.jitter, fold.width)

    def test_stride(self):
        fold = TrainSequenceFolder(
            self.seq_len,
            self.pad_id,
            self.int_overlap,
            self.jitter
        )
        self.assertEqual(self.seq_len - self.int_overlap, fold.stride)


class TestDefaultUsage(unittest.TestCase):

    def setUp(self):
        self.pad_id = -3
        self.fold = TrainSequenceFolder(7, self.pad_id)

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


class TestCustomUsage(unittest.TestCase):

    def setUp(self):
        self.seq_len = 11
        self.pad_id = -100
        self.overlap = 4
        self.jitter = 3
        self.fold = TrainSequenceFolder(
            self.seq_len,
            self.pad_id,
            self.overlap,
            self.jitter
        )

    def test_empty_shape(self):
        empty = pt.tensor([])
        actual = self.fold(empty)
        self.assertTupleEqual((1, 14), actual.shape)

    def test_empty_value(self):
        empty = pt.tensor([])
        actual = self.fold(empty)
        expected = self.pad_id * pt.ones(1, 14)
        pt.testing.assert_close(actual, expected)

    def test_one_element_shape(self):
        empty = pt.tensor([1])
        actual = self.fold(empty)
        self.assertTupleEqual((1, 14), actual.shape)

    def test_one_element_value(self):
        empty = pt.tensor([1.0])
        actual = self.fold(empty)
        expected = self.pad_id * pt.ones(1, 14)
        expected[:, 0] = 1.0
        pt.testing.assert_close(actual, expected)

    def test_single_max(self):
        single = pt.arange(1, 11)
        expected = pt.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + 4 * [self.pad_id]
        ])
        actual = self.fold(single)
        pt.testing.assert_close(actual, expected)

    def test_double_min(self):
        double = pt.arange(1, 12)
        expected = pt.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] + 3 * [self.pad_id],
            [8, 9, 10, 11] + 10 * [self.pad_id]
        ])
        actual = self.fold(double)
        pt.testing.assert_close(actual, expected)

    def test_double_max(self):
        double = pt.arange(1, 18)
        expected = pt.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17] + 4 * [self.pad_id]
        ])
        actual = self.fold(double)
        pt.testing.assert_close(actual, expected)

    def test_triple_min(self):
        triple = pt.arange(1, 19)
        expected = pt.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18] + 3 * [self.pad_id],
            [15, 16, 17, 18] + 10 * [self.pad_id]
        ])
        actual = self.fold(triple)
        pt.testing.assert_close(actual, expected)

    def test_triple_max(self):
        triple = pt.arange(1, 25)
        expected = pt.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            [15, 16, 17, 18, 19, 20, 21, 22, 23, 24] + 4 * [self.pad_id]
        ])
        actual = self.fold(triple)
        pt.testing.assert_close(actual, expected)

    def test_quadruple_min(self):
        quadruple = pt.arange(1, 26)
        expected = pt.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] + 3 * [self.pad_id],
            [22, 23, 24, 25] + 10 * [self.pad_id]
        ])
        actual = self.fold(quadruple)
        pt.testing.assert_close(actual, expected)

    def test_quadruple_max(self):
        quadruple = pt.arange(1, 32)
        expected = pt.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
            [22, 23, 24, 25, 26, 27, 28, 29, 30, 31] + 4 * [self.pad_id]
        ])
        actual = self.fold(quadruple)
        pt.testing.assert_close(actual, expected)

    def test_quintuple_min(self):
        quintuple = pt.arange(1, 33)
        expected = pt.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
            [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] + 3 * [self.pad_id],
            [29, 30, 31, 32] + 10 * [self.pad_id]
        ])
        actual = self.fold(quintuple)
        pt.testing.assert_close(actual, expected)

    def test_quintuple_max(self):
        quintuple = pt.arange(1, 39)
        expected = pt.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
            [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
            [29, 30, 31, 32, 33, 34, 35, 36, 37, 38] + 4 * [self.pad_id]
        ])
        actual = self.fold(quintuple)
        pt.testing.assert_close(actual, expected)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        fold = TrainSequenceFolder(512)
        expected = 'TrainSequenceFolder(512, 0, 0, 1)'
        self.assertEqual(expected, repr(fold))

    def test_custom_repr(self):
        fold = TrainSequenceFolder(512, -100, 0.25, 32)
        expected = 'TrainSequenceFolder(512, -100, 128, 32)'
        self.assertEqual(expected, repr(fold))

    def test_pickle_works(self):
        fold = TrainSequenceFolder(512, -100, 0.25, 32)
        _ = pickle.loads(pickle.dumps(fold))


if __name__ == '__main__':
    unittest.main()
