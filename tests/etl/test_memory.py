import pickle
import unittest
import ctypes
from unittest.mock import patch, Mock
from slangmod.etl import MemoryTrimmer


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.trim = MemoryTrimmer()

    def test_has_cdll(self):
        self.assertTrue(hasattr(self.trim, 'cdll'))

    def test_cdll(self):
        self.assertEqual('libc.so.6', self.trim.cdll)

    def test_has_libc(self):
        self.assertTrue(hasattr(self.trim, 'libc'))

    def test_libc(self):
        self.assertIsInstance(self.trim.libc, ctypes.CDLL)
        self.assertEqual('libc.so.6', self.trim.libc._name)


class TestCustomAttributes(unittest.TestCase):

    def setUp(self):
        self.trim = MemoryTrimmer('libc.so')

    def test_cdll(self):
        self.assertEqual('libc.so', self.trim.cdll)

    def test_cdll_stripped(self):
        trim = MemoryTrimmer('  libc.so ')
        self.assertEqual('libc.so', trim.cdll)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.trim = MemoryTrimmer()

    @patch('slangmod.etl.memory.CDLL')
    def test_no_arg(self, _):
        actual = self.trim()
        self.assertTupleEqual((), actual)

    @patch('slangmod.etl.memory.CDLL')
    def test_one_arg(self, _):
        arg = object()
        actual = self.trim(arg)
        self.assertIs(arg, actual)

    @patch('slangmod.etl.memory.CDLL')
    def test_multiple_args(self, _):
        args = object(), object(), object()
        actual = self.trim(*args)
        self.assertTupleEqual(args, actual)

    @patch('slangmod.etl.memory.CDLL')
    def test_default_library_loaded_once(self, mock):
        trim = MemoryTrimmer()
        mock.assert_not_called()
        _ = trim()
        _ = trim()
        _ = trim()
        mock.assert_called_once_with(trim.cdll)

    @patch('slangmod.etl.memory.CDLL')
    def test_custom_library_loaded_once(self, mock):
        trim = MemoryTrimmer('cdll')
        mock.assert_not_called()
        _ = trim()
        _ = trim()
        _ = trim()
        mock.assert_called_once_with('cdll')

    @patch('slangmod.etl.memory.CDLL')
    def test_malloc_trim_called(self, mock):
        library = Mock()
        mock.return_value = library
        _ = self.trim()
        library.malloc_trim.assert_called_once_with(0)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        trim = MemoryTrimmer()
        expected = "MemoryTrimmer('libc.so.6')"
        self.assertEqual(expected, repr(trim))

    def test_custom_repr(self):
        trim = MemoryTrimmer('libc.so')
        expected = "MemoryTrimmer('libc.so')"
        self.assertEqual(expected, repr(trim))

    def test_pickle_works(self):
        trim = MemoryTrimmer()
        _ = pickle.loads(pickle.dumps(trim))


if __name__ == '__main__':
    unittest.main()
