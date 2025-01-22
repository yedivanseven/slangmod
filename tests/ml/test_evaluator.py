import pickle
import unittest
from unittest.mock import Mock
import torch as pt
from slangmod.ml import Evaluator


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.pad_id = 3
        self.loss = pt.nn.CrossEntropyLoss(ignore_index=self.pad_id)
        self.batch_size = 4
        self.eval = Evaluator(self.loss, self.batch_size)

    def test_has_loss(self):
        self.assertTrue(hasattr(self.eval, 'loss'))

    def test_loss(self):
        self.assertIs(self.eval.loss, self.loss)

    def test_has_batch_size(self):
        self.assertTrue(hasattr(self.eval, 'batch_size'))

    def test_batch_size(self):
        self.assertIsInstance(self.eval.batch_size, int)
        self.assertEqual(self.batch_size, self.eval.batch_size)

    def test_has_show_progress(self):
        self.assertTrue(hasattr(self.eval, 'show_progress'))

    def test_show_progress(self):
        self.assertIsInstance(self.eval.show_progress, bool)
        self.assertTrue(self.eval.show_progress)

    def test_has_pad_id(self):
        self.assertTrue(hasattr(self.eval, 'pad_id'))

    def test_pad_id(self):
        self.assertIsInstance(self.eval.pad_id, int)
        self.assertEqual(self.pad_id, self.eval.pad_id)

    def test_has_top(self):
        self.assertTrue(hasattr(self.eval, 'top'))

    def test_has_perplexity(self):
        self.assertTrue(hasattr(self.eval, 'perplexity'))

    def test_raises_on_wrong_reduction(self):
        loss = pt.nn.CrossEntropyLoss(reduction='none')
        with self.assertRaises(ValueError):
            _ = Evaluator(loss, self.batch_size)


class TestTopK(unittest.TestCase):

    def setUp(self):
        self.pad_id = 1
        self.loss = pt.nn.CrossEntropyLoss(ignore_index=self.pad_id)
        self.batch_size = 2
        self.vocab = 13
        self.seq_len = 15
        self.logits = pt.rand(
            self.batch_size,
            self.vocab,
            self.seq_len,
            device='cpu'
        )
        self.eval = Evaluator(self.loss, self.batch_size)

    def test_callable(self):
        self.assertTrue(callable(self.eval.top))

    def test_top_1_accuracy_100(self):
        targets = self.logits.argmax(dim=1)
        n_tokens = (targets != self.pad_id).sum()
        n_hits = self.eval.top(1, self.logits, targets)
        self.assertEqual(n_tokens, n_hits)

    def test_top_2_accuracy_100(self):
        targets = self.logits.argmax(dim=1)
        n_tokens = (targets != self.pad_id).sum()
        n_hits = self.eval.top(2, self.logits, targets)
        self.assertEqual(n_tokens, n_hits)

    def test_top_5_accuracy_100(self):
        targets = self.logits.argmax(dim=1)
        n_tokens = (targets != self.pad_id).sum()
        n_hits = self.eval.top(5, self.logits, targets)
        self.assertEqual(n_tokens, n_hits)

    def test_top_1_accuracy_0(self):
        targets = self.logits.argmax(dim=1) + self.vocab
        n_hits = self.eval.top(1, self.logits, targets)
        self.assertEqual(0, n_hits)

    def test_top_2_accuracy_0(self):
        targets = self.logits.argmax(dim=1) + self.vocab
        n_hits = self.eval.top(2, self.logits, targets)
        self.assertEqual(0, n_hits)

    def test_top_5_accuracy_0(self):
        targets = self.logits.argmax(dim=1) + self.vocab
        n_hits = self.eval.top(5, self.logits, targets)
        self.assertEqual(0, n_hits)

    def test_top_1_accuracy_pad_id_ignored(self):
        targets = self.logits.argmax(dim=1) + self.vocab
        targets[0, 0] = self.pad_id
        targets[self.batch_size - 1, self.seq_len - 1] = self.pad_id
        self.logits[0, self.pad_id, 0] += 2.0
        self.logits[self.batch_size - 1, self.pad_id, self.seq_len - 1] += 2.0
        n_hits = self.eval.top(1, self.logits, targets)
        self.assertEqual(0, n_hits)

    def test_top_2_accuracy_pad_id_ignored(self):
        targets = self.logits.argmax(dim=1) + self.vocab
        targets[0, 0] = self.pad_id
        targets[self.batch_size - 1, self.seq_len - 1] = self.pad_id
        self.logits[0, self.pad_id, 0] += 2.0
        self.logits[self.batch_size - 1, self.pad_id, self.seq_len - 1] += 2.0
        n_hits = self.eval.top(2, self.logits, targets)
        self.assertEqual(0, n_hits)

    def test_top_5_accuracy_pad_id_ignored(self):
        targets = self.logits.argmax(dim=1) + self.vocab
        targets[0, 0] = self.pad_id
        targets[self.batch_size - 1, self.seq_len - 1] = self.pad_id
        self.logits[0, self.pad_id, 0] += 2.0
        self.logits[self.batch_size - 1, self.pad_id, self.seq_len - 1] += 2.0
        n_hits = self.eval.top(5, self.logits, targets)
        self.assertEqual(0, n_hits)

    def test_top_1_accuracy(self):
        targets = self.logits.argmax(dim=1) + self.vocab
        targets[0, 0] = 0
        targets[self.batch_size - 1, self.seq_len - 1] = 0
        self.logits[0, 0, 0] += 2.0
        self.logits[self.batch_size - 1, 0, self.seq_len - 1] += 2.0
        n_hits = self.eval.top(1, self.logits, targets)
        self.assertEqual(2, n_hits)

    def test_top_2_accuracy(self):
        targets = self.logits.argmax(dim=1) + self.vocab
        targets[0, 0] = 0
        targets[self.batch_size - 1, self.seq_len - 1] = 0
        self.logits[0, 0, 0] += 1.0
        self.logits[0, 2, 0] += 2.0
        self.logits[self.batch_size - 1, 0, self.seq_len - 1] += 1.0
        self.logits[self.batch_size - 1, 2, self.seq_len - 1] += 2.0
        n_hits = self.eval.top(1, self.logits, targets)
        self.assertEqual(0, n_hits)
        n_hits = self.eval.top(2, self.logits, targets)
        self.assertEqual(2, n_hits)

    def test_top_3_accuracy(self):
        targets = self.logits.argmax(dim=1) + self.vocab
        targets[0, 0] = 0
        targets[self.batch_size - 1, self.seq_len - 1] = 0
        self.logits[0, 0, 0] += 1.0
        self.logits[0, 2, 0] += 2.0
        self.logits[0, 3, 0] += 3.0
        self.logits[self.batch_size - 1, 0, self.seq_len - 1] += 1.0
        self.logits[self.batch_size - 1, 2, self.seq_len - 1] += 2.0
        self.logits[self.batch_size - 1, 3, self.seq_len - 1] += 3.0
        n_hits = self.eval.top(1, self.logits, targets)
        self.assertEqual(0, n_hits)
        n_hits = self.eval.top(2, self.logits, targets)
        self.assertEqual(0, n_hits)
        n_hits = self.eval.top(3, self.logits, targets)
        self.assertEqual(2, n_hits)


class TestPerplexity(unittest.TestCase):

    def setUp(self):
        self.pad_id = 1
        self.loss = pt.nn.CrossEntropyLoss(ignore_index=self.pad_id)
        self.batch_size = 2
        self.vocab = 13
        self.seq_len = 15
        self.logits = pt.rand(
            self.batch_size,
            self.vocab,
            self.seq_len,
            device='cpu'
        )
        self.eval = Evaluator(self.loss, self.batch_size)

    def test_callable(self):
        self.assertTrue(callable(self.eval.perplexity))

    def test_values(self):
        loss = pt.nn.CrossEntropyLoss(
            reduction='none',
            ignore_index=self.pad_id
        )
        targets = self.logits.argmax(dim=1)
        n_non_pad_tokens = (targets != self.pad_id).sum(dim=1)
        losses = loss(self.logits, targets).sum(dim=1) / n_non_pad_tokens
        expected = losses.exp().sum(dim=0)
        actual = self.eval.perplexity(self.logits, targets)
        pt.testing.assert_close(actual, expected)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.pad_id = 1
        self.loss = pt.nn.CrossEntropyLoss(ignore_index=self.pad_id)
        self.batch_size = 2
        self.vocab = 13
        self.seq_len = 15
        self.logits = pt.rand(
            self.batch_size,
            self.vocab,
            self.seq_len,
            device='cpu'
        )
        self.features = pt.randint(
            0,
            self.vocab,
            (self.batch_size, self.seq_len),
            device='cpu'
        )
        self.targets = self.logits.argmax(dim=1)
        self.eval = Evaluator(self.loss, self.batch_size)


    def test_callable(self):
        self.assertTrue(callable(self.eval))

    def test_data_sample_called(self):
        model = Mock(return_value=(self.logits,))
        data = Mock()
        data.sample.return_value = [((self.features, ), self.targets)]
        data.device = 'cpu'
        data.n = 2
        _ = self.eval(model, data)
        data.sample.assert_called_once_with(self.batch_size)

    def test_model_sample_called(self):
        model = Mock(return_value=(self.logits,))
        data = Mock()
        data.sample.return_value = [((self.features, ), self.targets)]
        data.device = 'cpu'
        data.n = 2
        _ = self.eval(model, data)
        model.assert_called_once_with(self.features)

    def test_return_type(self):
        model = Mock(return_value=(self.logits,))
        data = Mock()
        data.sample.return_value = [((self.features,), self.targets)]
        data.device = 'cpu'
        data.n = 2
        actual = self.eval(model, data)
        self.assertIsInstance(actual, tuple)
        self.assertEqual(5, len(actual))

    def test_returned_loss(self):
        model = Mock(return_value=(self.logits,))
        data = Mock()
        data.sample.return_value = [((self.features,), self.targets)]
        data.device = 'cpu'
        data.n = 2
        actual, *_ = self.eval(model, data)
        expected = self.loss(self.logits, self.targets)
        self.assertEqual(expected, actual)

    def test_returned_perplexity(self):
        model = Mock(return_value=(self.logits,))
        data = Mock()
        data.sample.return_value = [((self.features,), self.targets)]
        data.device = 'cpu'
        data.n = 2
        _, actual, *_ = self.eval(model, data)
        expected = self.eval.perplexity(self.logits, self.targets)
        self.assertEqual(expected / self.batch_size, actual)

    def test_returned_top_1(self):
        model = Mock(return_value=(self.logits,))
        data = Mock()
        data.sample.return_value = [((self.features,), self.targets)]
        data.device = 'cpu'
        data.n = 2
        _, _, actual, *_ = self.eval(model, data)
        expected = self.eval.top(1, self.logits, self.targets)
        n_tokens = (self.targets != self.pad_id).sum()
        self.assertEqual(expected / n_tokens, actual)

    def test_returned_top_2(self):
        model = Mock(return_value=(self.logits,))
        data = Mock()
        data.sample.return_value = [((self.features,), self.targets)]
        data.device = 'cpu'
        data.n = 2
        *_, actual, _ = self.eval(model, data)
        expected = self.eval.top(2, self.logits, self.targets)
        n_tokens = (self.targets != self.pad_id).sum()
        self.assertEqual(expected / n_tokens, actual)

    def test_returned_top_5(self):
        model = Mock(return_value=(self.logits,))
        data = Mock()
        data.sample.return_value = [((self.features,), self.targets)]
        data.device = 'cpu'
        data.n = 2
        *_, actual = self.eval(model, data)
        expected = self.eval.top(5, self.logits, self.targets)
        n_tokens = (self.targets != self.pad_id).sum()
        self.assertEqual(expected / n_tokens, actual)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.loss = pt.nn.CrossEntropyLoss()
        self.batch_size = 4

    def test_default_repr(self):
        evaluator = Evaluator(self.loss, self.batch_size)
        expected = 'Evaluator(CrossEntropyLoss(...), 4, True)'
        self.assertEqual(expected, repr(evaluator))

    def test_custom_repr(self):
        evaluator = Evaluator(self.loss, self.batch_size,False)
        expected = 'Evaluator(CrossEntropyLoss(...), 4, False)'
        self.assertEqual(expected, repr(evaluator))

    def test_pickle_works(self):
        evaluator = Evaluator(self.loss, self.batch_size, False)
        _ = pickle.loads(pickle.dumps(evaluator))


if __name__ == '__main__':
    unittest.main()
