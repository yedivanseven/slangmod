import pickle
import tomllib
import unittest
from unittest.mock import Mock
from slangmod.io.summary import TrainTomlPrinter


class TestAttributes(unittest.TestCase):

    def test_has_printer(self):
        train = TrainTomlPrinter()
        self.assertTrue(hasattr(train, 'printer'))

    def test_default_printer(self):
        train = TrainTomlPrinter()
        self.assertIs(train.printer, print)

    def test_custom_printer(self):
        obj = object()
        train = TrainTomlPrinter(obj)
        self.assertIs(train.printer, obj)

class TestUsage(unittest.TestCase):

    def setUp(self):
        self.history = {
            'train_loss': [0.11, 0.12],
            'test_loss': [0.21, 0.22],
            'lr': [0.01, 0.02]
        }

    def test_callable(self):
        train = TrainTomlPrinter()
        self.assertTrue(callable(train))

    def test_printer_called(self):
        printer = Mock()
        train = TrainTomlPrinter(printer)
        train(3, 2, 0.21, False, self.history)
        expected =(
            '\n[convergence]\n'
            'last_epoch = 2\n'
            'best_epoch = 1\n'
            'best_loss = 0.21\n'
            'max_epochs_reached = false\n'
            'train_loss = [\n'
            '    0.11,\n'
            '    0.12,\n'
            ']\n'
            'test_loss = [\n'
            '    0.21,\n'
            '    0.22,\n'
            ']\n'
            'learning_rate = [\n'
            '    0.01,\n'
            '    0.02,\n'
            ']\n'
        )
        printer.assert_called_once_with(expected)

    def test_msg_is_toml(self):
        printer = Mock()
        train = TrainTomlPrinter(printer)
        train(3, 2, 0.21, False, self.history)
        msg = printer.call_args[0][0]
        _ = tomllib.loads(msg)

    def test_toml_is_correct(self):
        printer = Mock()
        train = TrainTomlPrinter(printer)
        train(3, 2, 0.21, False, self.history)
        msg = printer.call_args[0][0]
        actual = tomllib.loads(msg)
        expected = {
            'convergence': {
                'last_epoch': 2,
                'best_epoch': 1,
                'best_loss': 0.21,
                'max_epochs_reached': False,
                'train_loss': [0.11, 0.12],
                'test_loss': [0.21, 0.22],
                'learning_rate': [0.01, 0.02]
            }
        }
        self.assertDictEqual(expected, actual)

    def test_nan_inf_values_string(self):
        printer = Mock()
        train = TrainTomlPrinter(printer)
        history = {
            'train_loss': [0.11, float('-inf')],
            'test_loss': [0.21, float('nan')],
            'lr': [0.01, float('inf')]
        }
        train(3, 2, 0.21, False, history)
        expected = (
            '\n[convergence]\n'
            'last_epoch = 2\n'
            'best_epoch = 1\n'
            'best_loss = 0.21\n'
            'max_epochs_reached = false\n'
            'train_loss = [\n'
            '    0.11,\n'
            '    -inf,\n'
            ']\n'
            'test_loss = [\n'
            '    0.21,\n'
            '    nan,\n'
            ']\n'
            'learning_rate = [\n'
            '    0.01,\n'
            '    inf,\n'
            ']\n'
        )
        printer.assert_called_once_with(expected)

    def test_nan_inf_values_toml(self):
        printer = Mock()
        train = TrainTomlPrinter(printer)
        history = {
            'train_loss': [0.11, float('-inf')],
            'test_loss': [0.21, float('inf')],
            'lr': [0.01, float('inf')]
        }
        train(3, 2, 0.21, False, history)
        msg = printer.call_args[0][0]
        actual = tomllib.loads(msg)
        expected = {
            'convergence': {
                'last_epoch': 2,
                'best_epoch': 1,
                'best_loss': 0.21,
                'max_epochs_reached': False,
                'train_loss': [0.11, float('-inf')],
                'test_loss': [0.21, float('inf')],
                'learning_rate': [0.01, float('inf')]
            }
        }
        self.assertDictEqual(expected, actual)


class TestMisc(unittest.TestCase):

    def test_repr(self):
        train = TrainTomlPrinter()
        expected = 'TrainTomlPrinter(print)'
        self.assertEqual(expected, repr(train))

    def test_pickle_works(self):
        train = TrainTomlPrinter()
        _ = pickle.loads(pickle.dumps(train))

    def test_pickle_raises_lambda(self):
        train = TrainTomlPrinter(lambda x: x)
        with self.assertRaises(AttributeError):
            _ = pickle.loads(pickle.dumps(train))


if __name__ == '__main__':
    unittest.main()
