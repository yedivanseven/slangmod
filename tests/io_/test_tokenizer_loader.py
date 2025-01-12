import pickle
import unittest
from unittest.mock import Mock
from tokenizers import Tokenizer
from tokenizers.models import BPE
from slangmod.io import TokenizerLoader


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer(BPE())
        self.path = 'path'
        self.load = TokenizerLoader(self.tokenizer, self.path)

    def test_has_algo(self):
        self.assertTrue(hasattr(self.load, 'algo'))

    def test_algo(self):
        self.assertIs(self.load.algo, self.tokenizer)

    def test_has_path(self):
        self.assertTrue(hasattr(self.load, 'path'))

    def test_path(self):
        self.assertEqual(self.path, self.load.path)

    def test_path_stringified(self):
        load = TokenizerLoader(self.tokenizer, 1)
        self.assertEqual('1', load.path)

    def test_path_stripped(self):
        load = TokenizerLoader(self.tokenizer, '  path ')
        self.assertEqual('path', load.path)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.algo = Mock()
        self.return_value = object()
        self.algo.from_file = Mock(return_value=self.return_value)

    def test_from_file_called_with_instantiation_path(self):
        load = TokenizerLoader(self.algo, '/tokenizer.json')
        actual = load()
        self.algo.from_file.assert_called_once_with('/tokenizer.json')
        self.assertIs(actual, self.return_value)

    def test_from_file_called_with_call_path(self):
        load = TokenizerLoader(self.algo)
        actual = load('/tokenizer.json')
        self.algo.from_file.assert_called_once_with('/tokenizer.json')
        self.assertIs(actual, self.return_value)

    def test_from_file_called_with_partial_paths(self):
        load = TokenizerLoader(self.algo, '/path/to')
        actual = load('tokenizer.json')
        self.algo.from_file.assert_called_once_with('/path/to/tokenizer.json')
        self.assertIs(actual, self.return_value)



class TestMisc(unittest.TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer(BPE())
        self.path = 'path'
        self.load = TokenizerLoader(self.tokenizer, self.path)

    def test_repr(self):
        expected = "TokenizerLoader('path')"
        self.assertEqual(expected, repr(self.load))

    def test_pickle_works(self):
        _ = pickle.loads(pickle.dumps(self.load))


if __name__ == '__main__':
    unittest.main()
