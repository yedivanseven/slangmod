import pickle
import unittest
from hashlib import sha256
from unittest.mock import patch, Mock
from pandas import Series, DataFrame, testing
from slangmod.etl import CorpusCleaner


def proc(doc: str) -> str:
    return doc.upper()


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.clean = CorpusCleaner(proc)

    def test_has_process(self):
        self.assertTrue(hasattr(self.clean, 'process'))

    def test_process(self):
        self.assertIs(self.clean.process, proc)

    def test_has_args(self):
        self.assertTrue(hasattr(self.clean, 'args'))

    def test_args(self):
        self.assertTupleEqual((), self.clean.args)

    def test_has_kwargs(self):
        self.assertTrue(hasattr(self.clean, 'kwargs'))

    def test_kwargs(self):
        self.assertDictEqual({}, self.clean.kwargs)


class TestCustomAttributes(unittest.TestCase):

    def setUp(self):
        self.args = 'docs',
        self.kwargs = {'leave': False}
        self.clean = CorpusCleaner(proc, *self.args, **self.kwargs)

    def test_args(self):
        self.assertTupleEqual(self.args, self.clean.args)

    def test_kwargs(self):
        self.assertDictEqual(self.kwargs, self.clean.kwargs)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.args = 'docs',
        self.kwargs = {'leave': False}
        self.name = 'text'
        self.corpus = Series(['hello', 'world'])

    def test_callable(self):
        clean = CorpusCleaner(proc)
        self.assertTrue(callable(clean))

    @patch('slangmod.etl.cleaner.tqdm')
    def test_tqdm_called_default(self, mock):
        mock.return_value = self.corpus
        clean = CorpusCleaner(proc)
        _ = clean(self.corpus)
        mock.assert_called_once_with(self.corpus)

    @patch('slangmod.etl.cleaner.tqdm')
    def test_tqdm_called_custom(self, mock):
        mock.return_value = self.corpus
        clean = CorpusCleaner(proc, *self.args, **self.kwargs)
        _ = clean(self.corpus)
        mock.assert_called_once_with(self.corpus, *self.args, **self.kwargs)

    @patch('slangmod.etl.cleaner.tqdm')
    def test_kwargs_merged(self, mock):
        mock.return_value = self.corpus
        kwargs = {'leave': True, 'total': 42}
        merged_kwargs = self.kwargs | kwargs
        clean = CorpusCleaner(proc, *self.args, **kwargs)
        _ = clean(self.corpus)
        mock.assert_called_once_with(self.corpus, *self.args, **merged_kwargs)

    def test_process_called(self):
        mock = Mock(return_value='42')
        clean = CorpusCleaner(mock)
        _ = clean(self.corpus)
        self.assertEqual(len(self.corpus), mock.call_count)

    def test_return_value_types(self):
        clean = CorpusCleaner(proc)
        actual = clean(self.corpus)
        self.assertIsInstance(actual, tuple)
        self.assertEqual(2, len(actual))
        frame, name = actual
        self.assertIsInstance(frame, DataFrame)
        self.assertIsInstance(name, str)

    def test_returned_frame_shape(self):
        clean = CorpusCleaner(proc)
        actual, _ = clean(self.corpus)
        self.assertTupleEqual((len(self.corpus), 1), actual.shape)

    def test_returned_frame_content(self):
        clean = CorpusCleaner(proc)
        actual, _ = clean(self.corpus)
        expected = Series([self.corpus[0].upper(), self.corpus[1].upper()])
        testing.assert_frame_equal(actual, expected.to_frame())

    def test_returned_frame_column_names(self):
        self.corpus.name = 'text'
        clean = CorpusCleaner(proc)
        actual, _ = clean(self.corpus)
        self.assertEqual('text', actual.columns[0])

    def test_returned_hash(self):
        clean = CorpusCleaner(proc)
        _, actual = clean(self.corpus)
        series = Series([self.corpus[0].upper(), self.corpus[1].upper()])
        expected = sha256(str(series).encode()).hexdigest()
        self.assertEqual(expected, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        clean = CorpusCleaner(proc)
        expected = 'CorpusCleaner(proc)'
        self.assertEqual(expected, repr(clean))

    def test_custom_repr(self):
        clean = CorpusCleaner(proc, 'docs', leave=False)
        expected = "CorpusCleaner(proc, 'docs', leave=False)"
        self.assertEqual(expected, repr(clean))

    def test_pickle_works(self):
        clean = CorpusCleaner(proc, 'docs', leave=False)
        _ = pickle.dumps(clean)


if __name__ == '__main__':
    unittest.main()
