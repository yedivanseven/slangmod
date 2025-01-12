import pickle
import itertools
import unittest
from unittest.mock import Mock
from slangmod.io import CorpusLoader


def reader(file: str) -> list[str]:
    return [file, file]


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.load = CorpusLoader(reader)

    def test_has_reader(self):
        self.assertTrue(hasattr(self.load, 'reader'))

    def test_reader(self):
        self.assertIs(reader, self.load.reader)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.files = ['file1', 'file2']
        self.expected = ['file1', 'file1', 'file2', 'file2']

    def test_callable(self):
        load = CorpusLoader(reader)
        self.assertTrue(callable(load))

    def test_empty_return_type(self):
        load = CorpusLoader(reader)
        actual = load([])
        self.assertIsInstance(actual, itertools.chain)

    def test_empty_return_value(self):
        load = CorpusLoader(reader)
        actual = load([])
        self.assertListEqual([], list(actual))

    def test_empty_reader_not_called(self):
        mock = Mock()
        load = CorpusLoader(mock)
        _ = load([])
        mock.assert_not_called()

    def test_return_type(self):
        load = CorpusLoader(reader)
        actual = load(self.files)
        self.assertIsInstance(actual, itertools.chain)

    def test_return_value(self):
        load = CorpusLoader(reader)
        actual = load(self.files)
        self.assertListEqual(self.expected, list(actual))

    def test_reader_called(self):
        mock = Mock(return_value=['mock'])
        load = CorpusLoader(mock)
        actual = load(self.files)
        _ = list(actual)
        self.assertEqual(2, mock.call_count)
        self.assertEqual(self.files[0], mock.call_args_list[0][0][0])
        self.assertEqual(self.files[1], mock.call_args_list[1][0][0])


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.load = CorpusLoader(reader)

    def test_repr(self):
        expected = "CorpusLoader(reader)"
        self.assertEqual(expected, repr(self.load))

    def tet_pickle_works(self):
        _ = pickle.dumps(self.load)


if __name__ == '__main__':
    unittest.main()
