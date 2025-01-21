import pickle
import unittest
from unittest.mock import patch
from collections.abc import Iterator
import torch as pt
from slangmod.ml import TrainData


class TestDefaultAttributes(unittest.TestCase):

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
        self.train = TrainData(self.seqs)

    def test_has_seqs(self):
        self.assertTrue(hasattr(self.train, 'seqs'))

    def test_seqs(self):
        self.assertIs(self.train.seqs, self.seqs)

    def test_has_shuffle(self):
        self.assertTrue(hasattr(self.train, 'shuffle'))

    def test_shuffle(self):
        self.assertIsInstance(self.train.shuffle, bool)
        self.assertTrue(self.train.shuffle)

    def test_has_jitter(self):
        self.assertTrue(hasattr(self.train, 'jitter'))

    def test_jitter(self):
        self.assertIsInstance(self.train.jitter, int)
        self.assertEqual(1, self.train.jitter)

    def test_has_device(self):
        self.assertTrue(hasattr(self.train, 'device'))

    def test_device(self):
        self.assertEqual(self.device, self.train.device)

    def test_has_n(self):
        self.assertTrue(hasattr(self.train, 'n'))

    def test_n(self):
        self.assertIsInstance(self.train.n, int)
        self.assertEqual(self.n, self.train.n)

    def test_has_seq_len(self):
        self.assertTrue(hasattr(self.train, 'seq_len'))

    def test_seq_len(self):
        self.assertIsInstance(self.train.seq_len, int)
        self.assertEqual(self.seq_len, self.train.seq_len)

    def test_has_sample(self):
        self.assertTrue(hasattr(self.train, 'sample'))

    def test_sample(self):
        self.assertTrue(callable(self.train.sample))


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.n = 7
        self.seq_len = 17
        self.device = pt.device('cpu')
        self.jitter = 3
        self.shuffle = False
        self.seqs = pt.randint(
            0,
            13,
            (self.n, self.seq_len + self.jitter),
            device=self.device,
            dtype=pt.long
        )
        self.train = TrainData(
            self.seqs,
            self.shuffle,
            self.jitter,
            device=self.device
        )

    def test_shuffle(self):
        self.assertFalse(self.train.shuffle)

    def test_jitter(self):
        self.assertEqual(3, self.train.jitter)

    def test_seq_len(self):
        self.assertEqual(self.seq_len, self.train.seq_len)

    def test_device(self):
        self.assertEqual(self.device, self.train.device)

    def test_max_jitter(self):
        _ = TrainData(
            self.seqs,
            self.shuffle,
            self.seq_len + self.jitter - 2,
            device=self.device
        )

    def test_jitter_too_high_raises(self):
        with self.assertRaises(ValueError):
            _ = TrainData(
                self.seqs,
                self.shuffle,
                self.seq_len + self.jitter - 1,
                device=self.device
            )


class TestDefaultSample(unittest.TestCase):

    def setUp(self):
        self.n = 7
        self.seq_len = 17
        self.jitter = 3
        self.device = pt.device('cpu')
        self.seqs = pt.randint(
            0,
            13,
            (self.n, self.seq_len + self.jitter),
            device=self.device,
            dtype=pt.long
        )
        self.train = TrainData(
            self.seqs,
            jitter=self.jitter,
            device=self.device
        )

    def test_return_type(self):
        batches = self.train.sample(4)
        self.assertIsInstance(batches, Iterator)

    def test_number_of_batches(self):
        batches = self.train.sample(4)
        n_batches = 0
        for _ in batches:
            n_batches += 1
        self.assertEqual(2, n_batches)

    def test_batch_structure(self):
        batches = self.train.sample(4)
        for batch in batches:
            inp, tgt = batch
            src, *_, is_causal = inp

    def test_first_batch_values(self):
        batches = self.train.sample(4)
        first = next(batches)
        inp, tgt = first
        src, *_, is_causal = inp
        self.assertIsInstance(is_causal, bool)
        self.assertTrue(is_causal)
        pt.testing.assert_close(src, self.seqs[:4, :self.seq_len])
        pt.testing.assert_close(tgt, self.seqs[:4, 1:self.seq_len + 1])

    def test_second_batch_values(self):
        batches = self.train.sample(4)
        _ = next(batches)
        second = next(batches)
        inp, tgt = second
        src, *_, is_causal = inp
        self.assertIsInstance(is_causal, bool)
        self.assertTrue(is_causal)
        pt.testing.assert_close(src, self.seqs[4:, :self.seq_len])
        pt.testing.assert_close(tgt, self.seqs[4:, 1:self.seq_len + 1])


class TestMaxNSampling(unittest.TestCase):

    def setUp(self):
        self.n = 7
        self.seq_len = 17
        self.jitter = 3
        self.device = pt.device('cpu')
        self.seqs = pt.randint(
            0,
            13,
            (self.n, self.seq_len + self.jitter),
            device=self.device,
            dtype=pt.long
        )
        self.train = TrainData(
            self.seqs,
            jitter=self.jitter,
            device=self.device
        )

    def test_number_of_batches_max_n_too_large(self):
        batches = self.train.sample(4, 100)
        n_batches = 0
        for _ in batches:
            n_batches += 1
        self.assertEqual(2, n_batches)

    def test_first_batch_size_max_n_too_large(self):
        batches = self.train.sample(4, 100)
        first = next(batches)
        inp, tgt = first
        src, *_, is_causal = inp
        self.assertEqual(4, src.size(0))
        self.assertEqual(4, tgt.size(0))

    def test_second_batch_size_max_n_too_large(self):
        batches = self.train.sample(4, 100)
        _ = next(batches)
        second = next(batches)
        inp, tgt = second
        src, *_, is_causal = inp
        self.assertEqual(3, src.size(0))
        self.assertEqual(3, tgt.size(0))

    def test_number_of_batches_max_n_clips_n(self):
        batches = self.train.sample(4, 5)
        n_batches = 0
        for _ in batches:
            n_batches += 1
        self.assertEqual(2, n_batches)

    def test_first_batch_size_max_n_clips_n(self):
        batches = self.train.sample(4, 5)
        first = next(batches)
        inp, tgt = first
        src, *_, is_causal = inp
        self.assertEqual(4, src.size(0))
        self.assertEqual(4, tgt.size(0))

    def test_second_batch_size_max_n_clips_n(self):
        batches = self.train.sample(4, 5)
        _ = next(batches)
        second = next(batches)
        inp, tgt = second
        src, *_, is_causal = inp
        self.assertEqual(3, src.size(0))
        self.assertEqual(3, tgt.size(0))

    def test_number_of_batches_max_n_clips_n_batches(self):
        batches = self.train.sample(4, 3)
        n_batches = 0
        for _ in batches:
            n_batches += 1
        self.assertEqual(1, n_batches)

    def test_first_batch_size_max_n_clips_n_batches(self):
        batches = self.train.sample(4, 3)
        first = next(batches)
        inp, tgt = first
        src, *_, is_causal = inp
        self.assertEqual(4, src.size(0))
        self.assertEqual(4, tgt.size(0))


class TestDeterministicUsage(unittest.TestCase):

    def setUp(self):
        self.n = 7
        self.seq_len = 17
        self.jitter = 3
        self.device = pt.device('cpu')
        self.seqs = pt.randint(
            0,
            13,
            (self.n, self.seq_len + self.jitter),
            device=self.device,
            dtype=pt.long
        )
        self.train = TrainData(
            self.seqs,
            False,
            jitter=self.jitter,
            device=self.device
        )

    def test_callable(self):
        self.assertTrue(callable(self.train))

    def test_return_type(self):
        n_batches, batches = self.train(4)
        self.assertIsInstance(n_batches, int)
        self.assertIsInstance(batches, Iterator)

    def test_number_of_batches(self):
        returned, batches = self.train(4)
        actual = 0
        for _ in batches:
            actual+= 1
        self.assertEqual(2, returned)
        self.assertEqual(2, actual)

    def test_batch_structure(self):
        _, batches = self.train(4)
        for batch in batches:
            inp, tgt = batch
            src, *_, is_causal = inp

    def test_first_batch_values(self):
        _, batches = self.train(4)
        first = next(batches)
        inp, tgt = first
        src, *_, is_causal = inp
        self.assertIsInstance(is_causal, bool)
        self.assertTrue(is_causal)
        pt.testing.assert_close(src, self.seqs[:4, :self.seq_len])
        pt.testing.assert_close(tgt, self.seqs[:4, 1:self.seq_len + 1])

    def test_second_batch_values(self):
        _, batches = self.train(4)
        _ = next(batches)
        second = next(batches)
        inp, tgt = second
        src, *_, is_causal = inp
        self.assertIsInstance(is_causal, bool)
        self.assertTrue(is_causal)
        pt.testing.assert_close(src, self.seqs[4:, :self.seq_len])
        pt.testing.assert_close(tgt, self.seqs[4:, 1:self.seq_len + 1])

    def test_n_batches_step_freq_2(self):
        n_batches, _ = self.train(2, 2)
        self.assertEqual(2, n_batches)

    def test_first_batch_step_freq_2(self):
        _, batches = self.train(2, 2)
        first = next(batches)
        inp, tgt = first
        src, *_, is_causal = inp
        self.assertTrue(is_causal)
        pt.testing.assert_close(src, self.seqs[:2, :self.seq_len])
        pt.testing.assert_close(tgt, self.seqs[:2, 1:self.seq_len + 1])

    def test_second_batch_step_freq_2(self):
        _, batches = self.train(2, 2)
        _ = next(batches)
        second = next(batches)
        inp, tgt = second
        src, *_, is_causal = inp
        self.assertTrue(is_causal)
        pt.testing.assert_close(src, self.seqs[2:4, :self.seq_len])
        pt.testing.assert_close(tgt, self.seqs[2:4, 1:self.seq_len + 1])

    def test_n_batches_step_freq_3(self):
        n_batches, _ = self.train(2, 3)
        self.assertEqual(3, n_batches)

    def test_first_batch_step_freq_3(self):
        _, batches = self.train(2, 3)
        first = next(batches)
        inp, tgt = first
        src, *_, is_causal = inp
        self.assertTrue(is_causal)
        pt.testing.assert_close(src, self.seqs[:2, :self.seq_len])
        pt.testing.assert_close(tgt, self.seqs[:2, 1:self.seq_len + 1])

    def test_second_batch_step_freq_3(self):
        _, batches = self.train(2, 3)
        _ = next(batches)
        second = next(batches)
        inp, tgt = second
        src, *_, is_causal = inp
        self.assertTrue(is_causal)
        pt.testing.assert_close(src, self.seqs[2:4, :self.seq_len])
        pt.testing.assert_close(tgt, self.seqs[2:4, 1:self.seq_len + 1])

    def test_third_batch_step_freq_3(self):
        _, batches = self.train(2, 3)
        _ = next(batches)
        _ = next(batches)
        third = next(batches)
        inp, tgt = third
        src, *_, is_causal = inp
        self.assertTrue(is_causal)
        pt.testing.assert_close(src, self.seqs[4:6, :self.seq_len])
        pt.testing.assert_close(tgt, self.seqs[4:6, 1:self.seq_len + 1])

    @patch('torch.randint', return_value=2)
    @patch('torch.randperm', return_value = [1, 0])
    def test_no_randomization_called(self, randperm, randint):
        _ = self.train(2, 3)
        randperm.assert_not_called()
        randint.assert_not_called()


class TestRandomizeUsage(unittest.TestCase):

    def setUp(self):
        self.n = 7
        self.seq_len = 17
        self.jitter = 3
        self.device = pt.device('cpu')
        self.seqs = pt.randint(
            0,
            13,
            (self.n, self.seq_len + self.jitter),
            device=self.device,
            dtype=pt.long
        )
        self.batch_size = 4
        self.train = TrainData(
            self.seqs,
            True,
            jitter=self.jitter,
            device=self.device
        )

    @patch('torch.randint', return_value=pt.tensor([1], device='cpu'))
    @patch('torch.randperm', return_value=pt.tensor([1, 0], device='cpu'))
    def test_randomization_called(self, randperm, randint):
        _ = self.train(self.batch_size)
        randperm.assert_called_once_with(2, device=self.seqs.device)
        randint.assert_called_once_with(
            0,
            self.jitter,
            [1],
            device=self.seqs.device
        )

    @patch('torch.randint', return_value=pt.tensor([1], device='cpu'))
    @patch('torch.randperm', return_value=pt.tensor([1, 0], device='cpu'))
    def test_first_batch_values_start_1(self, _, __):
        _, batches = self.train(self.batch_size)
        first = next(batches)
        inp, tgt = first
        src, *_, is_causal = inp
        self.assertTrue(is_causal)
        pt.testing.assert_close(src, self.seqs[4:, 1:1 + self.seq_len])
        pt.testing.assert_close(tgt, self.seqs[4:, 2:2 + self.seq_len])

    @patch('torch.randint', return_value=pt.tensor([1], device='cpu'))
    @patch('torch.randperm', return_value=pt.tensor([1, 0], device='cpu'))
    def test_second_batch_values_start_1(self, _, __):
        _, batches = self.train(self.batch_size)
        _ = next(batches)
        second = next(batches)
        inp, tgt = second
        src, *_, is_causal = inp
        self.assertTrue(is_causal)
        pt.testing.assert_close(src, self.seqs[:4, 1:1 + self.seq_len])
        pt.testing.assert_close(tgt, self.seqs[:4, 2:2 + self.seq_len])

    @patch('torch.randint', return_value=pt.tensor([2], device='cpu'))
    @patch('torch.randperm', return_value=pt.tensor([1, 0], device='cpu'))
    def test_first_batch_values_start_2(self, _, __):
        _, batches = self.train(self.batch_size)
        first = next(batches)
        inp, tgt = first
        src, *_, is_causal = inp
        self.assertTrue(is_causal)
        pt.testing.assert_close(src, self.seqs[4:, 2:2 + self.seq_len])
        pt.testing.assert_close(tgt, self.seqs[4:, 3:3 + self.seq_len])

    @patch('torch.randint', return_value=pt.tensor([2], device='cpu'))
    @patch('torch.randperm', return_value=pt.tensor([1, 0], device='cpu'))
    def test_second_batch_values_start_2(self, _, __):
        _, batches = self.train(self.batch_size)
        _ = next(batches)
        second = next(batches)
        inp, tgt = second
        src, *_, is_causal = inp
        self.assertTrue(is_causal)
        pt.testing.assert_close(src, self.seqs[:4, 2:2 + self.seq_len])
        pt.testing.assert_close(tgt, self.seqs[:4, 3:3 + self.seq_len])


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.n = 7
        self.seq_len = 17
        self.jitter = 3
        self.device = pt.device('cpu')
        self.seqs = pt.randint(
            0,
            13,
            (self.n, self.seq_len + self.jitter),
            device=self.device,
            dtype=pt.long
        )
        self.train = TrainData(
            self.seqs,
            jitter=self.jitter,
            device=self.device
        )

    def test_repr(self):
        expected = f'TrainData(n={self.n})'
        self.assertEqual(expected, repr(self.train))

    def test_pickle_works(self):
        _ = pickle.loads(pickle.dumps(self.train))


if __name__ == '__main__':
    unittest.main()
