import pickle
import unittest
from collections.abc import Iterator
import torch as pt
from slangmod.ml import TestData


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.n = 7
        self.seq_len = 17
        self.device = pt.device('cpu')
        self.seqs = pt.randint(
            0,
            13,
            (self.n, self.seq_len + 1),
            device=self.device,
            dtype=pt.long
        )
        self.test = TestData(self.seqs, device=self.device)

    def test_has_seqs(self):
        self.assertTrue(hasattr(self.test, 'seqs'))

    def test_seqs(self):
        self.assertIs(self.test.seqs, self.seqs)

    def test_has_device(self):
        self.assertTrue(hasattr(self.test, 'device'))

    def test_device(self):
        self.assertEqual(self.device, self.test.device)

    def test_has_n(self):
        self.assertTrue(hasattr(self.test, 'n'))

    def test_n(self):
        self.assertIsInstance(self.test.n, int)
        self.assertEqual(self.n, self.test.n)

    def test_has_seq_len(self):
        self.assertTrue(hasattr(self.test, 'seq_len'))

    def test_seq_len(self):
        self.assertIsInstance(self.test.seq_len, int)
        self.assertEqual(self.seq_len, self.test.seq_len)

    def test_has_sample(self):
        self.assertTrue(hasattr(self.test, 'sample'))

    def test_sample(self):
        self.assertTrue(callable(self.test.sample))


class TestDefaultUsage(unittest.TestCase):

    def setUp(self):
        self.n = 7
        self.seq_len = 17
        self.device = pt.device('cpu')
        self.seqs = pt.randint(
            0,
            13,
            (self.n, self.seq_len + 1),
            device=self.device,
            dtype=pt.long
        )
        self.test = TestData(self.seqs, device=self.device)

    def test_return_type(self):
        batches = self.test.sample(4)
        self.assertIsInstance(batches, Iterator)

    def test_number_of_batches(self):
        batches = self.test.sample(4)
        n_batches = 0
        for _ in batches:
            n_batches += 1
        self.assertEqual(2, n_batches)

    def test_batch_structure(self):
        batches = self.test.sample(4)
        for batch in batches:
            inp, tgt = batch
            src, = inp

    def test_first_batch_values(self):
        batches = self.test.sample(4)
        first = next(batches)
        inp, tgt = first
        src, = inp
        pt.testing.assert_close(src, self.seqs[:4, :-1])
        pt.testing.assert_close(tgt, self.seqs[:4, 1:])

    def test_second_batch_values(self):
        batches = self.test.sample(4)
        _ = next(batches)
        second = next(batches)
        inp, tgt = second
        src, = inp
        pt.testing.assert_close(src, self.seqs[4:, :-1])
        pt.testing.assert_close(tgt, self.seqs[4:, 1:])


class TestMaxN(unittest.TestCase):

    def setUp(self):
        self.n = 7
        self.seq_len = 17
        self.device = pt.device('cpu')
        self.seqs = pt.randint(
            0,
            13,
            (self.n, self.seq_len + 1),
            device=self.device,
            dtype=pt.long
        )
        self.test = TestData(self.seqs, device=self.device)

    def test_number_of_batches_max_n_too_large(self):
        batches = self.test.sample(4, 100)
        n_batches = 0
        for _ in batches:
            n_batches += 1
        self.assertEqual(2, n_batches)

    def test_first_batch_size_max_n_too_large(self):
        batches = self.test.sample(4, 100)
        first = next(batches)
        inp, tgt = first
        src, = inp
        self.assertEqual(4, src.size(0))
        self.assertEqual(4, tgt.size(0))

    def test_second_batch_size_max_n_too_large(self):
        batches = self.test.sample(4, 100)
        _ = next(batches)
        second = next(batches)
        inp, tgt = second
        src, = inp
        self.assertEqual(3, src.size(0))
        self.assertEqual(3, tgt.size(0))

    def test_number_of_batches_max_n_clips_n(self):
        batches = self.test.sample(4, 5)
        n_batches = 0
        for _ in batches:
            n_batches += 1
        self.assertEqual(2, n_batches)

    def test_first_batch_size_max_n_clips_n(self):
        batches = self.test.sample(4, 5)
        first = next(batches)
        inp, tgt = first
        src, = inp
        self.assertEqual(4, src.size(0))
        self.assertEqual(4, tgt.size(0))

    def test_second_batch_size_max_n_clips_n(self):
        batches = self.test.sample(4, 5)
        _ = next(batches)
        second = next(batches)
        inp, tgt = second
        src, = inp
        self.assertEqual(1, src.size(0))
        self.assertEqual(1, tgt.size(0))

    def test_number_of_batches_max_n_clips_n_batches(self):
        batches = self.test.sample(4, 3)
        n_batches = 0
        for _ in batches:
            n_batches += 1
        self.assertEqual(1, n_batches)

    def test_first_batch_size_max_n_clips_n_batches(self):
        batches = self.test.sample(4, 3)
        first = next(batches)
        inp, tgt = first
        src, = inp
        self.assertEqual(3, src.size(0))
        self.assertEqual(3, tgt.size(0))


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.n = 7
        self.seq_len = 17
        self.device = pt.device('cpu')
        self.seqs = pt.randint(
            0,
            13,
            (self.n, self.seq_len + 1),
            device=self.device,
            dtype=pt.long
        )
        self.test = TestData(self.seqs, device=self.device)

    def test_repr(self):
        expected = f'TestData(n={self.n})'
        self.assertEqual(expected, repr(self.test))

    def test_pickle_works(self):
        _ = pickle.loads(pickle.dumps(self.test))


if __name__ == '__main__':
    unittest.main()
