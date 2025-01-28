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
            '\n[training]\n'
            'last_epoch = 2\n'
            'best_epoch = 1\n'
            'best_loss = 0.21000\n'
            'max_epochs_reached = false\n'
            '\n[[training.epochs]]\n'
            'train_loss = 0.11000\n'
            'test_loss = 0.21000\n'
            'learning_rate = 0.01000\n'
            '\n[[training.epochs]]\n'
            'train_loss = 0.12000\n'
            'test_loss = 0.22000\n'
            'learning_rate = 0.02000\n'
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
            'training': {
                'last_epoch': 2,
                'best_epoch': 1,
                'best_loss': 0.21,
                'max_epochs_reached': False,
                'epochs': [
                    {
                        'train_loss': 0.11,
                        'test_loss': 0.21,
                        'learning_rate': 0.01
                    },
                    {
                        'train_loss': 0.12,
                        'test_loss': 0.22,
                        'learning_rate': 0.02
                    },
                ]
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
