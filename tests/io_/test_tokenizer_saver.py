import pickle
import unittest
from unittest.mock import Mock
from pathlib import Path
from tempfile import TemporaryDirectory
from slangmod.io import TokenizerSaver
from tokenizers import Tokenizer
from tokenizers.models import BPE


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.path = 'path'
        self.save = TokenizerSaver(self.path)

    def test_has_path(self):
        self.assertTrue(hasattr(self.save, 'path'))

    def test_path(self):
        self.assertEqual(self.path, self.save.path)

    def test_path_stringified(self):
        save = TokenizerSaver(1)
        self.assertEqual('1', save.path)

    def test_path_stripped(self):
        save = TokenizerSaver('  path ')
        self.assertEqual('path', save.path)

    def test_has_create(self):
        self.assertTrue(hasattr(self.save, 'path'))

    def test_create(self):
        self.assertIsInstance(self.save.create, bool)
        self.assertFalse(self.save.create)


class TestCustomAttributes(unittest.TestCase):

    def setUp(self):
        self.create = True
        self.save = TokenizerSaver('path', self.create)

    def test_create(self):
        self.assertIsInstance(self.save.create, bool)
        self.assertTrue(self.save.create)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.folder = TemporaryDirectory()
        self.file = 'tokenizer.json'
        self.path = self.folder.name + '/' + self.file
        self.subdir = 'subdir'
        self.subpath = self.folder.name + '/' + self.subdir + '/' + self.file
        self.tokenizer = Tokenizer(BPE())

    def tearDown(self):
        self.folder.cleanup()
        del self.folder

    def test_callable(self):
        save = TokenizerSaver('path')
        self.assertTrue(callable(save))

    def test_mock_save_instantiation_path_dir_exists_create_false(self):
        save = TokenizerSaver(self.path)
        algo = Mock()
        actual = save(algo)
        algo.save.asssert_called_once_with(self.path)
        self.assertTupleEqual((), actual)

    def test_save_instantiation_path_dir_exists_create_false(self):
        save = TokenizerSaver(self.path)
        actual = save(self.tokenizer)
        self.assertTrue(Path(self.path).exists())
        self.assertTrue(Path(self.path).is_file())
        self.assertTupleEqual((), actual)

    def test_mock_save_instantiation_path_dir_exists_create_true(self):
        save = TokenizerSaver(self.path, True)
        algo = Mock()
        actual = save(algo)
        algo.save.asssert_called_once_with(self.path)
        self.assertTupleEqual((), actual)

    def test_save_instantiation_path_dir_exists_create_true(self):
        save = TokenizerSaver(self.path, True)
        actual = save(self.tokenizer)
        self.assertTrue(Path(self.path).exists())
        self.assertTrue(Path(self.path).is_file())
        self.assertTupleEqual((), actual)

    def test_mock_save_instantiation_path_dir_not_exists_create_false(self):
        save = TokenizerSaver(self.subpath)
        algo = Mock()
        self.assertFalse(Path(self.subpath).parent.exists())
        actual = save(algo)
        self.assertFalse(Path(self.subpath).parent.exists())
        algo.save.asssert_called_once_with(self.subpath)
        self.assertTupleEqual((), actual)

    def test_save_instantiation_path_dir_not_exists_create_false(self):
        save = TokenizerSaver(self.subpath)
        self.assertFalse(Path(self.subpath).parent.exists())
        with self.assertRaises(Exception):
            _ = save(self.tokenizer)
        self.assertFalse(Path(self.subpath).parent.exists())

    def test_mock_save_instantiation_path_dir_not_exists_create_true(self):
        save = TokenizerSaver(self.subpath, True)
        algo = Mock()
        self.assertFalse(Path(self.subpath).parent.exists())
        actual = save(algo)
        self.assertTrue(Path(self.subpath).parent.exists())
        algo.save.asssert_called_once_with(self.subpath)
        self.assertTupleEqual((), actual)

    def test_save_instantiation_path_dir_not_exists_create_true(self):
        save = TokenizerSaver(self.subpath, True)
        self.assertFalse(Path(self.subpath).parent.exists())
        actual = save(self.tokenizer)
        self.assertTrue(Path(self.subpath).exists())
        self.assertTrue(Path(self.subpath).is_file())
        self.assertTupleEqual((), actual)

    def test_mock_interpolate_parts(self):
        template = self.folder.name + '/{}/{}'
        save = TokenizerSaver(template, True)
        self.assertFalse(Path(self.subpath).parent.exists())
        algo = Mock()
        actual = save(algo, self.subdir, self.file)
        self.assertTrue(Path(self.subpath).parent.exists())
        algo.save.asssert_called_once_with(self.subpath)
        self.assertTupleEqual((), actual)

    def test_interpolate_parts(self):
        template = self.folder.name + '/{}/{}'
        save = TokenizerSaver(template, True)
        self.assertFalse(Path(self.subpath).parent.exists())
        actual = save(self.tokenizer, self.subdir, self.file)
        self.assertTrue(Path(self.subpath).exists())
        self.assertTrue(Path(self.subpath).is_file())
        self.assertTupleEqual((), actual)


class TestMisc(unittest.TestCase):

    def test_default_repr(self):
        save = TokenizerSaver('path')
        expected = "TokenizerSaver('path', False)"
        self.assertEqual(expected, repr(save))

    def test_custom_repr(self):
        save = TokenizerSaver('path', True)
        expected = "TokenizerSaver('path', True)"
        self.assertEqual(expected, repr(save))

    def test_pickle_works(self):
        save = TokenizerSaver('path', True)
        _ = pickle.loads(pickle.dumps(save))


if __name__ == '__main__':
    unittest.main()
