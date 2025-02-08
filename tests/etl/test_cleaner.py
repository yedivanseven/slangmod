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

    def test_has_min_len(self):
        self.assertTrue(hasattr(self.clean, 'min_len'))

    def test_min_len(self):
        self.assertIsInstance(self.clean.min_len, int)
        self.assertEqual(1, self.clean.min_len)

    def test_has_show_progress(self):
        self.assertTrue(hasattr(self.clean, 'show_progress'))

    def test_show_progress(self):
        self.assertIsInstance(self.clean.show_progress, bool)
        self.assertTrue(self.clean.show_progress)


class TestCustomAttributes(unittest.TestCase):

    def setUp(self):
        self.min_len = 3
        self.show_progress = False
        self.clean = CorpusCleaner(proc, self.min_len, self.show_progress)

    def test_min_len(self):
        self.assertEqual(self.min_len, self.clean.min_len)

    def test_show_progress(self):
        self.assertFalse(self.clean.show_progress)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.min_len = 5
        self.name = 'text'
        self.data = ['hello', 'world', '42']
        self.corpus = Series(self.data, name=self.name)

    def test_callable(self):
        clean = CorpusCleaner(proc)
        self.assertTrue(callable(clean))

    @patch('slangmod.etl.cleaner.tqdm')
    def test_tqdm_called_default(self, mock):
        mock.return_value = self.corpus
        clean = CorpusCleaner(proc)
        _ = clean(self.corpus)
        mock.assert_called_once_with(self.corpus, 'Documents', disable=False)

    @patch('slangmod.etl.cleaner.tqdm')
    def test_tqdm_called_custom(self, mock):
        mock.return_value = self.corpus
        clean = CorpusCleaner(proc, self.min_len, show_progress=False)
        _ = clean(self.corpus)
        mock.assert_called_once_with(self.corpus, 'Documents', disable=True)

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

    def test_returned_frame_shape_default(self):
        clean = CorpusCleaner(proc)
        actual, _ = clean(self.corpus)
        self.assertTupleEqual((len(self.data), 1), actual.shape)

    def test_returned_frame_shape_custom(self):
        clean = CorpusCleaner(proc, self.min_len)
        actual, _ = clean(self.corpus)
        self.assertTupleEqual((len(self.data) - 1, 1), actual.shape)

    def test_returned_frame_content_default(self):
        clean = CorpusCleaner(proc)
        actual, _ = clean(self.corpus)
        expected = Series([doc.upper() for doc in self.data], name=self.name)
        testing.assert_frame_equal(actual, expected.to_frame())

    def test_returned_frame_content_custom(self):
        clean = CorpusCleaner(proc, self.min_len)
        actual, _ = clean(self.corpus)
        expected = Series(
            [doc.upper() for doc in self.data[:-1]],
            name=self.name
        )
        testing.assert_frame_equal(actual, expected.to_frame())

    def test_returned_frame_column_names(self):
        clean = CorpusCleaner(proc)
        actual, _ = clean(self.corpus)
        self.assertEqual(self.name, actual.columns[0])

    def test_returned_hash_default(self):
        clean = CorpusCleaner(proc)
        _, actual = clean(self.corpus)
        series = Series([doc.upper() for doc in self.data], name=self.name)
        expected = sha256(str(series).encode()).hexdigest()
        self.assertEqual(expected, actual)

    def test_returned_hash_custom(self):
        clean = CorpusCleaner(proc, self.min_len)
        _, actual = clean(self.corpus)
        series = Series(
            [doc.upper() for doc in self.data[:-1]],
            name=self.name
        )
        expected = sha256(str(series).encode()).hexdigest()
        self.assertEqual(expected, actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        clean = CorpusCleaner(proc)
        expected = 'CorpusCleaner(proc, 1, True)'
        self.assertEqual(expected, repr(clean))

    def test_custom_repr(self):
        clean = CorpusCleaner(proc, 5,False)
        expected = "CorpusCleaner(proc, 5, False)"
        self.assertEqual(expected, repr(clean))

    def test_pickle_works(self):
        clean = CorpusCleaner(proc, 3,False)
        _ = pickle.loads(pickle.dumps(clean))


if __name__ == '__main__':
    unittest.main()
