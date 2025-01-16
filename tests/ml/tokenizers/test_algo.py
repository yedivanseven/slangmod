import pickle
import unittest
from unittest.mock import patch
from pandas import Series
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece, WordLevel
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Digits
from tokenizers.decoders import CTC
from tokenizers.processors import ByteLevel
from tokenizers.trainers import (
    BpeTrainer,
    UnigramTrainer,
    WordPieceTrainer,
    WordLevelTrainer
)
from slangmod.ml.tokenizers import Special
from slangmod.ml import Algo


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.pad = AddedToken('[PAD]', special=True)
        self.unk = AddedToken('[UNK]', special=True)
        self.eos = AddedToken('[EOS]', special=True)
        self.cls = AddedToken('[CLS]', special=True)
        self.mask = AddedToken('[MASK]', special=True)
        self.special = Special(
            [self.cls],
            self.mask,
            pad=self.pad,
            unk=self.unk,
            eos=self.eos
        )
        self.model = BPE(unk_token=self.unk.content)
        self.tokenizer = Tokenizer(self.model)
        self.algo = Algo(self.special, self.tokenizer)

    def test_has_special(self):
        self.assertTrue(hasattr(self.algo, 'special'))

    def test_special(self):
        self.assertIs(self.algo.special, self.special)

    def test_has_tokenizer(self):
        self.assertTrue(hasattr(self.algo, 'tokenizer'))

    def test_has_trainer(self):
        self.assertTrue(hasattr(self.algo, 'trainer'))

    def test_has_vocab(self):
        self.assertTrue(hasattr(self.algo, 'vocab'))

    def test_has_unk_id(self):
        self.assertTrue(hasattr(self.algo, 'unk_id'))

    def test_unk_id(self):
        self.assertIsInstance(self.algo.unk_id, int)
        self.assertEqual(1, self.algo.unk_id)

    def test_has_eos_id(self):
        self.assertTrue(hasattr(self.algo, 'eos_id'))

    def test_eos_id(self):
        self.assertIsInstance(self.algo.eos_id, int)
        self.assertEqual(3, self.algo.eos_id)

    def test_callable(self):
        self.assertTrue(callable(self.algo))

    def test_has_from_buffer(self):
        self.assertTrue(hasattr(self.algo, 'from_buffer'))

    def test_from_buffer_callable(self):
        self.assertTrue(callable(self.algo.from_buffer))

    def test_has_from_file(self):
        self.assertTrue(hasattr(self.algo, 'from_file'))

    def test_from_file_callable(self):
        self.assertTrue(callable(self.algo.from_file))

    def test_has_from_pretrained(self):
        self.assertTrue(hasattr(self.algo, 'from_pretrained'))

    def test_from_pretrained_callable(self):
        self.assertTrue(callable(self.algo.from_pretrained))

    def test_has_from_str(self):
        self.assertTrue(hasattr(self.algo, 'from_str'))

    def test_from_str_callable(self):
        self.assertTrue(callable(self.algo.from_str))

    def test_has_train(self):
        self.assertTrue(hasattr(self.algo, 'train'))

    def test_train_callable(self):
        self.assertTrue(callable(self.algo.train))

    def test_has_train_from_iterator(self):
        self.assertTrue(hasattr(self.algo, 'train_from_iterator'))

    def test_train_from_iterator_callable(self):
        self.assertTrue(callable(self.algo.train_from_iterator))

    def test_has_terminate(self):
        self.assertTrue(hasattr(self.algo, 'terminate'))

    def test_terminate_callable(self):
        self.assertTrue(callable(self.algo.terminate))


class TestBPETokenizer(unittest.TestCase):

    def setUp(self):
        self.pad = AddedToken('[PAD]', special=True)
        self.unk = AddedToken('[UNK]', special=True)
        self.eos = AddedToken('[EOS]', special=True)
        self.cls = AddedToken('[CLS]', special=True)
        self.mask = AddedToken('[MASK]', special=True)
        self.special = Special(
            [self.cls],
            self.mask,
            pad=self.pad,
            unk=self.unk,
            eos=self.eos
        )

    def test_tokenizer(self):
        model = BPE()
        tokenizer = Tokenizer(model)
        algo = Algo(self.special, tokenizer)
        self.assertIs(algo.tokenizer, tokenizer)

    def test_model(self):
        model = BPE()
        tokenizer = Tokenizer(model)
        algo = Algo(self.special, tokenizer)
        self.assertIsInstance(algo.tokenizer.model, BPE)

    def test_model_cast_to_tokenizer(self):
        model = BPE()
        algo = Algo(self.special, model)
        self.assertIsInstance(algo.tokenizer, Tokenizer)
        self.assertIsInstance(algo.tokenizer.model, BPE)

    def test_special_tokens_added(self):
        model = BPE()
        algo = Algo(self.special, model)
        self.assertDictEqual(
            self.special.decoder,
            algo.tokenizer.get_added_tokens_decoder()
        )

    def test_special_tokens_fully_overwritten(self):
        model = BPE()
        tokenizer = Tokenizer(model)
        tokenizer.add_special_tokens(self.special.tokens)
        algo = Algo(self.special, tokenizer)
        self.assertDictEqual(
            self.special.decoder,
            algo.tokenizer.get_added_tokens_decoder()
        )

    def test_special_tokens_extended(self):
        model = BPE()
        tokenizer = Tokenizer(model)
        tokenizer.add_special_tokens([self.pad, self.unk])
        algo = Algo(self.special, tokenizer)
        self.assertDictEqual(
            self.special.decoder,
            algo.tokenizer.get_added_tokens_decoder()
        )

    def test_more_special_tokens_allowed(self):
        model = BPE()
        tokenizer = Tokenizer(model)
        tokens = [self.pad, self.unk, self.eos, self.mask, self.cls]
        tokenizer.add_special_tokens(tokens)
        special = Special(pad=self.pad, unk=self.unk, eos=self.eos)
        algo = Algo(special, tokenizer)
        self.assertDictEqual(
            dict(enumerate(tokens)),
            algo.tokenizer.get_added_tokens_decoder()
        )

    def test_model_unk_token_set(self):
        model = BPE()
        self.assertIsNone(model.unk_token)
        algo = Algo(self.special, model)
        self.assertEqual(self.unk.content, algo.tokenizer.model.unk_token)

    def test_wrong_token_order_raises(self):
        model = BPE()
        tokenizer = Tokenizer(model)
        tokenizer.add_special_tokens([self.unk, self.pad, self.eos])
        special = Special(pad=self.pad, unk=self.unk, eos=self.eos)
        with self.assertRaises(ValueError):
            _ = Algo(special, tokenizer)

    def test_wrong_token_raises(self):
        model = BPE()
        tokenizer = Tokenizer(model)
        tokenizer.add_special_tokens([self.eos])
        special = Special(pad=self.pad, unk=self.unk, eos=self.eos)
        with self.assertRaises(ValueError):
            _ = Algo(special, tokenizer)

    def test_wrong_unk_token_in_model_raises(self):
        model = BPE(unk_token='wrong')
        self.assertEqual('wrong', model.unk_token)
        with self.assertRaises(ValueError):
            _ = Algo(self.special, model)

    def test_wrong_unk_token_in_tokenizer_raises(self):
        model = BPE(unk_token='wrong')
        tokenizer = Tokenizer(model)
        self.assertEqual('wrong', model.unk_token)
        with self.assertRaises(ValueError):
            _ = Algo(self.special, tokenizer)


class TestBPETrainer(unittest.TestCase):

    def setUp(self):
        self.pad = AddedToken('[PAD]', special=True)
        self.unk = AddedToken('[UNK]', special=True)
        self.eos = AddedToken('[EOS]', special=True)
        self.cls = AddedToken('[CLS]', special=True)
        self.mask = AddedToken('[MASK]', special=True)
        self.special = Special(
            [self.cls],
            self.mask,
            pad=self.pad,
            unk=self.unk,
            eos=self.eos
        )
        self.model = BPE(unk_token=self.unk.content)
        self.tokenizer = Tokenizer(self.model)

    def test_default_trainer(self):
        algo = Algo(self.special, self.tokenizer)
        self.assertIsInstance(algo.trainer, BpeTrainer)

    def test_default_trainer_special_tokens(self):
        algo = Algo(self.special, self.tokenizer)
        self.assertListEqual(self.special.tokens, algo.trainer.special_tokens)

    def test_default_trainer_vocab_size(self):
        algo = Algo(self.special, self.tokenizer)
        self.assertEqual(algo.trainer.vocab_size, algo.vocab)

    def test_correct_special_tokens_work(self):
        trainer = BpeTrainer(special_tokens=self.special.tokens)
        algo = Algo(self.special, self.tokenizer, trainer)
        self.assertListEqual(self.special.tokens, algo.trainer.special_tokens)

    def test_custom_trainer_vocab_size(self):
        trainer = BpeTrainer(
            vocab_size=1234,
            special_tokens=self.special.tokens
        )
        algo = Algo(self.special, self.tokenizer, trainer)
        self.assertEqual(1234, algo.vocab)

    def test_default_special_string_tokens_work(self):
        tokens = [
            self.pad.content,
            self.unk.content,
            self.cls.content,
            self.eos.content,
            self.mask.content
        ]
        trainer = BpeTrainer(special_tokens=tokens)
        algo = Algo(self.special, self.tokenizer, trainer)
        self.assertListEqual(self.special.tokens, algo.trainer.special_tokens)

    def test_wrong_trainer_raises(self):
        trainer = WordPieceTrainer(special_tokens=self.special.tokens)
        with self.assertRaises(TypeError):
            _ = Algo(self.special, self.tokenizer, trainer)

    def test_wrong_token_order_raises(self):
        tokens = [self.pad, self.unk, self.eos, self.cls, self.mask]
        trainer = BpeTrainer(special_tokens=tokens)
        with self.assertRaises(ValueError):
            _ = Algo(self.special, self.tokenizer, trainer)

    def test_wrong_token_number_raises(self):
        tokens = [self.pad, self.unk, self.eos]
        trainer = BpeTrainer(special_tokens=tokens)
        with self.assertRaises(ValueError):
            _ = Algo(self.special, self.tokenizer, trainer)

    def test_wrong_token_type_raises(self):
        cls = AddedToken('[CLS]', special=True, lstrip=True)
        tokens = [self.pad, self.unk, cls, self.eos, self.mask]
        trainer = BpeTrainer(special_tokens=tokens)
        with self.assertRaises(ValueError):
            _ = Algo(self.special, self.tokenizer, trainer)


class TestWordpieceTokenizer(unittest.TestCase):

    def setUp(self):
        self.pad = AddedToken('[PAD]', special=True)
        self.unk = AddedToken('<unk>', special=True)
        self.eos = AddedToken('[EOS]', special=True)
        self.cls = AddedToken('[CLS]', special=True)
        self.mask = AddedToken('[MASK]', special=True)
        self.special = Special(
            [self.cls],
            self.mask,
            pad=self.pad,
            unk=self.unk,
            eos=self.eos
        )

    def test_tokenizer(self):
        model = WordPiece(unk_token=self.unk.content)
        tokenizer = Tokenizer(model)
        algo = Algo(self.special, tokenizer)
        self.assertIs(algo.tokenizer, tokenizer)

    def test_model(self):
        model = WordPiece(unk_token=self.unk.content)
        tokenizer = Tokenizer(model)
        algo = Algo(self.special, tokenizer)
        self.assertIsInstance(algo.tokenizer.model, WordPiece)

    def test_model_cast_to_tokenizer(self):
        model = WordPiece(unk_token=self.unk.content)
        algo = Algo(self.special, model)
        self.assertIsInstance(algo.tokenizer, Tokenizer)
        self.assertIsInstance(algo.tokenizer.model, WordPiece)

    def test_special_tokens_added(self):
        model = WordPiece(unk_token=self.unk.content)
        algo = Algo(self.special, model)
        self.assertDictEqual(
            self.special.decoder,
            algo.tokenizer.get_added_tokens_decoder()
        )

    def test_special_tokens_fully_overwritten(self):
        model = WordPiece(unk_token=self.unk.content)
        tokenizer = Tokenizer(model)
        tokenizer.add_special_tokens(self.special.tokens)
        algo = Algo(self.special, tokenizer)
        self.assertDictEqual(
            self.special.decoder,
            algo.tokenizer.get_added_tokens_decoder()
        )

    def test_special_tokens_extended(self):
        model = WordPiece(unk_token=self.unk.content)
        tokenizer = Tokenizer(model)
        tokenizer.add_special_tokens([self.pad, self.unk])
        algo = Algo(self.special, tokenizer)
        self.assertDictEqual(
            self.special.decoder,
            algo.tokenizer.get_added_tokens_decoder()
        )

    def test_more_special_tokens_allowed(self):
        model = WordPiece(unk_token=self.unk.content)
        tokenizer = Tokenizer(model)
        tokens = [self.pad, self.unk, self.eos, self.mask, self.cls]
        tokenizer.add_special_tokens(tokens)
        special = Special(pad=self.pad, unk=self.unk, eos=self.eos)
        algo = Algo(special, tokenizer)
        self.assertDictEqual(
            dict(enumerate(tokens)),
            algo.tokenizer.get_added_tokens_decoder()
        )

    def test_wrong_token_order_raises(self):
        model = WordPiece(unk_token=self.unk.content)
        tokenizer = Tokenizer(model)
        tokenizer.add_special_tokens([self.unk, self.pad, self.eos])
        special = Special(pad=self.pad, unk=self.unk, eos=self.eos)
        with self.assertRaises(ValueError):
            _ = Algo(special, tokenizer)

    def test_wrong_token_raises(self):
        model = WordPiece(unk_token=self.unk.content)
        tokenizer = Tokenizer(model)
        tokenizer.add_special_tokens([self.eos])
        special = Special(pad=self.pad, unk=self.unk, eos=self.eos)
        with self.assertRaises(ValueError):
            _ = Algo(special, tokenizer)

    def test_wrong_unk_token_in_model_raises(self):
        model = WordPiece(unk_token='wrong')
        self.assertEqual('wrong', model.unk_token)
        with self.assertRaises(ValueError):
            _ = Algo(self.special, model)

    def test_wrong_unk_token_in_tokenizer_raises(self):
        model = WordPiece(unk_token='wrong')
        tokenizer = Tokenizer(model)
        self.assertEqual('wrong', model.unk_token)
        with self.assertRaises(ValueError):
            _ = Algo(self.special, tokenizer)


class TestWordPieceTrainer(unittest.TestCase):

    def setUp(self):
        self.pad = AddedToken('[PAD]', special=True)
        self.unk = AddedToken('<unk>', special=True)
        self.eos = AddedToken('[EOS]', special=True)
        self.cls = AddedToken('[CLS]', special=True)
        self.mask = AddedToken('[MASK]', special=True)
        self.special = Special(
            [self.cls],
            self.mask,
            pad=self.pad,
            unk=self.unk,
            eos=self.eos
        )
        self.model = WordPiece(unk_token=self.unk.content)
        self.tokenizer = Tokenizer(self.model)

    def test_default_trainer(self):
        algo = Algo(self.special, self.tokenizer)
        self.assertIsInstance(algo.trainer, WordPieceTrainer)

    def test_default_trainer_special_tokens(self):
        algo = Algo(self.special, self.tokenizer)
        self.assertListEqual(self.special.tokens, algo.trainer.special_tokens)

    def test_default_trainer_vocab_size(self):
        algo = Algo(self.special, self.tokenizer)
        self.assertEqual(algo.trainer.vocab_size, algo.vocab)

    def test_correct_special_tokens_work(self):
        trainer = WordPieceTrainer(special_tokens=self.special.tokens)
        algo = Algo(self.special, self.tokenizer, trainer)
        self.assertListEqual(self.special.tokens, algo.trainer.special_tokens)

    def test_custom_trainer_vocab_size(self):
        trainer = WordPieceTrainer(
            vocab_size=1234,
            special_tokens=self.special.tokens
        )
        algo = Algo(self.special, self.tokenizer, trainer)
        self.assertEqual(1234, algo.vocab)

    def test_default_special_string_tokens_work(self):
        tokens = [
            self.pad.content,
            self.unk.content,
            self.cls.content,
            self.eos.content,
            self.mask.content
        ]
        trainer = WordPieceTrainer(special_tokens=tokens)
        algo = Algo(self.special, self.tokenizer, trainer)
        self.assertListEqual(self.special.tokens, algo.trainer.special_tokens)

    def test_wrong_trainer_raises(self):
        trainer = BpeTrainer(special_tokens=self.special.tokens)
        with self.assertRaises(TypeError):
            _ = Algo(self.special, self.tokenizer, trainer)

    def test_wrong_token_order_raises(self):
        tokens = [self.pad, self.unk, self.eos, self.cls, self.mask]
        trainer = WordPieceTrainer(special_tokens=tokens)
        with self.assertRaises(ValueError):
            _ = Algo(self.special, self.tokenizer, trainer)

    def test_wrong_token_number_raises(self):
        tokens = [self.pad, self.unk, self.eos]
        trainer = WordPieceTrainer(special_tokens=tokens)
        with self.assertRaises(ValueError):
            _ = Algo(self.special, self.tokenizer, trainer)

    def test_wrong_token_type_raises(self):
        cls = AddedToken('[CLS]', special=True, lstrip=True)
        tokens = [self.pad, self.unk, cls, self.eos, self.mask]
        trainer = WordPieceTrainer(special_tokens=tokens)
        with self.assertRaises(ValueError):
            _ = Algo(self.special, self.tokenizer, trainer)


class TestWordLevelTokenizer(unittest.TestCase):

    def setUp(self):
        self.pad = AddedToken('[PAD]', special=True)
        self.unk = AddedToken('[UNK]', special=True)
        self.eos = AddedToken('[EOS]', special=True)
        self.cls = AddedToken('[CLS]', special=True)
        self.mask = AddedToken('[MASK]', special=True)
        self.special = Special(
            [self.cls],
            self.mask,
            pad=self.pad,
            unk=self.unk,
            eos=self.eos
        )

    def test_tokenizer(self):
        model = WordLevel(unk_token=self.unk.content)
        tokenizer = Tokenizer(model)
        algo = Algo(self.special, tokenizer)
        self.assertIs(algo.tokenizer, tokenizer)

    def test_model(self):
        model = WordLevel(unk_token=self.unk.content)
        tokenizer = Tokenizer(model)
        algo = Algo(self.special, tokenizer)
        self.assertIsInstance(algo.tokenizer.model, WordLevel)

    def test_model_cast_to_tokenizer(self):
        model = WordLevel(unk_token=self.unk.content)
        algo = Algo(self.special, model)
        self.assertIsInstance(algo.tokenizer, Tokenizer)
        self.assertIsInstance(algo.tokenizer.model, WordLevel)

    def test_special_tokens_added(self):
        model = WordLevel(unk_token=self.unk.content)
        algo = Algo(self.special, model)
        self.assertDictEqual(
            self.special.decoder,
            algo.tokenizer.get_added_tokens_decoder()
        )

    def test_special_tokens_fully_overwritten(self):
        model = WordLevel(unk_token=self.unk.content)
        tokenizer = Tokenizer(model)
        tokenizer.add_special_tokens(self.special.tokens)
        algo = Algo(self.special, tokenizer)
        self.assertDictEqual(
            self.special.decoder,
            algo.tokenizer.get_added_tokens_decoder()
        )

    def test_special_tokens_extended(self):
        model = WordLevel(unk_token=self.unk.content)
        tokenizer = Tokenizer(model)
        tokenizer.add_special_tokens([self.pad, self.unk])
        algo = Algo(self.special, tokenizer)
        self.assertDictEqual(
            self.special.decoder,
            algo.tokenizer.get_added_tokens_decoder()
        )

    def test_more_special_tokens_allowed(self):
        model = WordLevel(unk_token=self.unk.content)
        tokenizer = Tokenizer(model)
        tokens = [self.pad, self.unk, self.eos, self.mask, self.cls]
        tokenizer.add_special_tokens(tokens)
        special = Special(pad=self.pad, unk=self.unk, eos=self.eos)
        algo = Algo(special, tokenizer)
        self.assertDictEqual(
            dict(enumerate(tokens)),
            algo.tokenizer.get_added_tokens_decoder()
        )

    def test_wrong_token_order_raises(self):
        model = WordLevel(unk_token=self.unk.content)
        tokenizer = Tokenizer(model)
        tokenizer.add_special_tokens([self.unk, self.pad, self.eos])
        special = Special(pad=self.pad, unk=self.unk, eos=self.eos)
        with self.assertRaises(ValueError):
            _ = Algo(special, tokenizer)

    def test_wrong_token_raises(self):
        model = WordLevel(unk_token=self.unk.content)
        tokenizer = Tokenizer(model)
        tokenizer.add_special_tokens([self.eos])
        special = Special(pad=self.pad, unk=self.unk, eos=self.eos)
        with self.assertRaises(ValueError):
            _ = Algo(special, tokenizer)

    def test_wrong_unk_token_in_model_raises(self):
        model = WordLevel(unk_token='wrong')
        self.assertEqual('wrong', model.unk_token)
        with self.assertRaises(ValueError):
            _ = Algo(self.special, model)

    def test_wrong_unk_token_in_tokenizer_raises(self):
        model = WordLevel(unk_token='wrong')
        tokenizer = Tokenizer(model)
        self.assertEqual('wrong', model.unk_token)
        with self.assertRaises(ValueError):
            _ = Algo(self.special, tokenizer)


class TestWordLevelTrainer(unittest.TestCase):

    def setUp(self):
        self.pad = AddedToken('[PAD]', special=True)
        self.unk = AddedToken('[UNK]', special=True)
        self.eos = AddedToken('[EOS]', special=True)
        self.cls = AddedToken('[CLS]', special=True)
        self.mask = AddedToken('[MASK]', special=True)
        self.special = Special(
            [self.cls],
            self.mask,
            pad=self.pad,
            unk=self.unk,
            eos=self.eos
        )
        self.model = WordLevel(unk_token=self.unk.content)
        self.tokenizer = Tokenizer(self.model)

    def test_default_trainer(self):
        algo = Algo(self.special, self.tokenizer)
        self.assertIsInstance(algo.trainer, WordLevelTrainer)

    def test_default_trainer_special_tokens(self):
        algo = Algo(self.special, self.tokenizer)
        self.assertListEqual(self.special.tokens, algo.trainer.special_tokens)

    def test_default_trainer_vocab_size(self):
        algo = Algo(self.special, self.tokenizer)
        self.assertEqual(algo.trainer.vocab_size, algo.vocab)

    def test_correct_special_tokens_work(self):
        trainer = WordLevelTrainer(special_tokens=self.special.tokens)
        algo = Algo(self.special, self.tokenizer, trainer)
        self.assertListEqual(self.special.tokens, algo.trainer.special_tokens)

    def test_custom_trainer_vocab_size(self):
        trainer = WordLevelTrainer(
            vocab_size=1234,
            special_tokens=self.special.tokens
        )
        algo = Algo(self.special, self.tokenizer, trainer)
        self.assertEqual(1234, algo.vocab)

    def test_default_special_string_tokens_work(self):
        tokens = [
            self.pad.content,
            self.unk.content,
            self.cls.content,
            self.eos.content,
            self.mask.content
        ]
        trainer = WordLevelTrainer(special_tokens=tokens)
        algo = Algo(self.special, self.tokenizer, trainer)
        self.assertListEqual(self.special.tokens, algo.trainer.special_tokens)

    def test_wrong_trainer_raises(self):
        trainer = BpeTrainer(special_tokens=self.special.tokens)
        with self.assertRaises(TypeError):
            _ = Algo(self.special, self.tokenizer, trainer)

    def test_wrong_token_order_raises(self):
        tokens = [self.pad, self.unk, self.eos, self.cls, self.mask]
        trainer = WordLevelTrainer(special_tokens=tokens)
        with self.assertRaises(ValueError):
            _ = Algo(self.special, self.tokenizer, trainer)

    def test_wrong_token_number_raises(self):
        tokens = [self.pad, self.unk, self.eos]
        trainer = WordLevelTrainer(special_tokens=tokens)
        with self.assertRaises(ValueError):
            _ = Algo(self.special, self.tokenizer, trainer)

    def test_wrong_token_type_raises(self):
        cls = AddedToken('[CLS]', special=True, lstrip=True)
        tokens = [self.pad, self.unk, cls, self.eos, self.mask]
        trainer = WordLevelTrainer(special_tokens=tokens)
        with self.assertRaises(ValueError):
            _ = Algo(self.special, self.tokenizer, trainer)


class TestUnigramTokenizer(unittest.TestCase):

    def setUp(self):
        self.pad = AddedToken('[PAD]', special=True)
        self.unk = AddedToken('[UNK]', special=True)
        self.eos = AddedToken('[EOS]', special=True)
        self.cls = AddedToken('[CLS]', special=True)
        self.mask = AddedToken('[MASK]', special=True)
        self.special = Special(
            [self.cls],
            self.mask,
            pad=self.pad,
            unk=self.unk,
            eos=self.eos
        )

    def test_tokenizer(self):
        model = Unigram(self.special.unigram_vocab, unk_id=self.special.unk_id)
        tokenizer = Tokenizer(model)
        algo = Algo(self.special, tokenizer)
        self.assertIs(algo.tokenizer, tokenizer)

    def test_model(self):
        model = Unigram(self.special.unigram_vocab, unk_id=self.special.unk_id)
        tokenizer = Tokenizer(model)
        algo = Algo(self.special, tokenizer)
        self.assertIsInstance(algo.tokenizer.model, Unigram)

    def test_model_cast_to_tokenizer(self):
        model = Unigram(self.special.unigram_vocab, unk_id=self.special.unk_id)
        algo = Algo(self.special, model)
        self.assertIsInstance(algo.tokenizer, Tokenizer)
        self.assertIsInstance(algo.tokenizer.model, Unigram)

    def test_special_tokens_added(self):
        model = Unigram(self.special.unigram_vocab, unk_id=self.special.unk_id)
        algo = Algo(self.special, model)
        self.assertDictEqual(
            self.special.decoder,
            algo.tokenizer.get_added_tokens_decoder()
        )

    def test_special_tokens_fully_overwritten(self):
        model = Unigram(self.special.unigram_vocab, unk_id=self.special.unk_id)
        tokenizer = Tokenizer(model)
        tokenizer.add_special_tokens(self.special.tokens)
        algo = Algo(self.special, tokenizer)
        self.assertDictEqual(
            self.special.decoder,
            algo.tokenizer.get_added_tokens_decoder()
        )

    def test_special_tokens_extended(self):
        model = Unigram(self.special.unigram_vocab, unk_id=self.special.unk_id)
        tokenizer = Tokenizer(model)
        tokenizer.add_special_tokens([self.pad, self.unk])
        algo = Algo(self.special, tokenizer)
        self.assertDictEqual(
            self.special.decoder,
            algo.tokenizer.get_added_tokens_decoder()
        )

    def test_more_special_tokens_allowed(self):
        special = Special(pad=self.pad, unk=self.unk, eos=self.eos)
        model = Unigram(special.unigram_vocab, unk_id=special.unk_id)
        tokenizer = Tokenizer(model)
        tokens = [self.pad, self.unk, self.eos, self.mask, self.cls]
        tokenizer.add_special_tokens(tokens)
        algo = Algo(special, tokenizer)
        self.assertDictEqual(
            dict(enumerate(tokens)),
            algo.tokenizer.get_added_tokens_decoder()
        )

    def test_wrong_unk_token_raises(self):
        model = Unigram(self.special.unigram_vocab, unk_id=0)
        with self.assertRaises(ValueError):
            _ = Algo(self.special, model)

    def test_wrong_token_order_raises(self):
        vocab = [
            (self.eos.content, 0.0),
            (self.unk.content, 0.0),
            (self.pad.content, 0.0)
        ]
        model = Unigram(vocab, unk_id=1)
        tokenizer = Tokenizer(model)
        special = Special(pad=self.pad, unk=self.unk, eos=self.eos)
        with self.assertRaises(ValueError):
            _ = Algo(special, tokenizer)

    def test_wrong_token_raises(self):
        vocab = [(self.unk.content, 0.0)]
        model = Unigram(vocab, unk_id=0)
        tokenizer = Tokenizer(model)
        special = Special(pad=self.pad, unk=self.unk, eos=self.eos)
        with self.assertRaises(ValueError):
            _ = Algo(special, tokenizer)

    def test_default_token_raises(self):
        model = Unigram()
        tokenizer = Tokenizer(model)
        special = Special(pad=self.pad, unk=self.unk, eos=self.eos)
        with self.assertRaises(ValueError):
            _ = Algo(special, tokenizer)

class TestUnigramTrainer(unittest.TestCase):

    def setUp(self):
        self.pad = AddedToken('[PAD]', special=True)
        self.unk = AddedToken('[UNK]', special=True)
        self.eos = AddedToken('[EOS]', special=True)
        self.cls = AddedToken('[CLS]', special=True)
        self.mask = AddedToken('[MASK]', special=True)
        self.special = Special(
            [self.cls],
            self.mask,
            pad=self.pad,
            unk=self.unk,
            eos=self.eos
        )
        self.model = Unigram(
            vocab=self.special.unigram_vocab,
            unk_id=self.special.unk_id
        )
        self.tokenizer = Tokenizer(self.model)

    def test_default_trainer(self):
        algo = Algo(self.special, self.tokenizer)
        self.assertIsInstance(algo.trainer, UnigramTrainer)

    def test_default_trainer_special_tokens(self):
        algo = Algo(self.special, self.tokenizer)
        self.assertListEqual(self.special.tokens, algo.trainer.special_tokens)

    def test_default_trainer_vocab_size(self):
        algo = Algo(self.special, self.tokenizer)
        self.assertEqual(algo.trainer.vocab_size, algo.vocab)

    def test_correct_special_tokens_work(self):
        trainer = UnigramTrainer(
            special_tokens=self.special.tokens,
            unk_token=self.special.unk.content
        )
        algo = Algo(self.special, self.tokenizer, trainer)
        self.assertListEqual(self.special.tokens, algo.trainer.special_tokens)

    def test_custom_trainer_vocab_size(self):
        trainer = UnigramTrainer(
            vocab_size=1234,
            special_tokens=self.special.tokens,
            unk_token=self.special.unk.content
        )
        algo = Algo(self.special, self.tokenizer, trainer)
        self.assertEqual(1234, algo.vocab)

    def test_default_special_string_tokens_work(self):
        tokens = [
            self.pad.content,
            self.unk.content,
            self.cls.content,
            self.eos.content,
            self.mask.content
        ]
        trainer = UnigramTrainer(
            special_tokens=tokens,
            unk_token=self.special.unk.content
        )
        algo = Algo(self.special, self.tokenizer, trainer)
        self.assertListEqual(self.special.tokens, algo.trainer.special_tokens)

    def test_wrong_trainer_raises(self):
        trainer = BpeTrainer(special_tokens=self.special.tokens)
        with self.assertRaises(TypeError):
            _ = Algo(self.special, self.tokenizer, trainer)

    def test_no_unk_token_raises(self):
        trainer = UnigramTrainer(special_tokens=self.special.tokens)
        with self.assertRaises(ValueError):
            _ = Algo(self.special, self.tokenizer, trainer)

    def test_wrong_unk_token_raised(self):
        trainer = UnigramTrainer(
            special_tokens=self.special.tokens,
            unk_token='wrong'
        )
        with self.assertRaises(ValueError):
            _ = Algo(self.special, self.tokenizer, trainer)

    def test_wrong_token_order_raises(self):
        tokens = [self.pad, self.unk, self.eos, self.cls, self.mask]
        trainer = UnigramTrainer(
            special_tokens=tokens,
            unk_token=self.special.unk.content
        )
        with self.assertRaises(ValueError):
            _ = Algo(self.special, self.tokenizer, trainer)

    def test_wrong_token_number_raises(self):
        tokens = [self.pad, self.unk, self.eos]
        trainer = UnigramTrainer(
            special_tokens=tokens,
            unk_token=self.special.unk.content
        )
        with self.assertRaises(ValueError):
            _ = Algo(self.special, self.tokenizer, trainer)

    def test_wrong_token_type_raises(self):
        cls = AddedToken('[CLS]', special=True, lstrip=True)
        tokens = [self.pad, self.unk, cls, self.eos, self.mask]
        trainer = UnigramTrainer(
            special_tokens=tokens,
            unk_token=self.special.unk.content
        )
        with self.assertRaises(ValueError):
            _ = Algo(self.special, self.tokenizer, trainer)


class TestOptionalArguments(unittest.TestCase):

    def setUp(self):
        self.pad = AddedToken('[PAD]', special=True)
        self.unk = AddedToken('[UNK]', special=True)
        self.eos = AddedToken('[EOS]', special=True)
        self.cls = AddedToken('[CLS]', special=True)
        self.mask = AddedToken('[MASK]', special=True)
        self.special = Special(
            [self.cls],
            self.mask,
            pad=self.pad,
            unk=self.unk,
            eos=self.eos
        )
        self.model = BPE(unk_token=self.unk.content)
        self.tokenizer = Tokenizer(self.model)

    def test_all_none_before_instantiation(self):
        self.assertIsNone(self.tokenizer.normalizer)
        self.assertIsNone(self.tokenizer.pre_tokenizer)
        self.assertIsNone(self.tokenizer.post_processor)
        self.assertIsNone(self.tokenizer.decoder)

    def test_all_none_after_instantiation(self):
        algo = Algo(self.special, self.tokenizer)
        self.assertIsNone(algo.tokenizer.normalizer)
        self.assertIsNone(algo.tokenizer.pre_tokenizer)
        self.assertIsNone(algo.tokenizer.post_processor)
        self.assertIsNone(algo.tokenizer.decoder)

    def test_normalizer_set(self):
        algo = Algo(self.special, self.tokenizer, normalizer=Lowercase())
        self.assertIsInstance(algo.tokenizer.normalizer, Lowercase)

    def test_pre_tokenizer_set(self):
        algo = Algo(self.special, self.tokenizer, pre_tokenizer=Digits())
        self.assertIsInstance(algo.tokenizer.pre_tokenizer, Digits)

    def test_decoder_set(self):
        algo = Algo(self.special, self.tokenizer, decoder=CTC())
        self.assertIsInstance(algo.tokenizer.decoder, CTC)

    def test_post_processor_set(self):
        algo = Algo(self.special, self.tokenizer, post_processor=ByteLevel())
        self.assertIsInstance(algo.tokenizer.post_processor, ByteLevel)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.pad = AddedToken('[PAD]', special=True)
        self.unk = AddedToken('[UNK]', special=True)
        self.eos = AddedToken('[EOS]', special=True)
        self.cls = AddedToken('[CLS]', special=True)
        self.mask = AddedToken('[MASK]', special=True)
        self.special = Special(
            [self.cls],
            self.mask,
            pad=self.pad,
            unk=self.unk,
            eos=self.eos
        )
        self.model = BPE(
            unk_token=self.unk.content,
            dropout=0.0,
            vocab=self.special.encoder,
            merges=[]
        )
        self.alternative = BPE(unk_token=self.unk.content, dropout=0.1)
        self.tokenizer = Tokenizer(self.model)
        self.trainer = BpeTrainer(
            vocab_size=1234,
            special_tokens=self.special.tokens
        )
        self.algo = Algo(self.special, self.tokenizer, self.trainer)

    @patch('slangmod.ml.tokenizers.algo.Tokenizer.from_buffer')
    def test_from_buffer_called(self, mock):
        expected = b'buffer'
        mock.return_value = self.alternative
        algo = self.algo.from_buffer(expected)
        mock.assert_called_once_with(expected)
        self.assertIsNot(algo, self.algo)
        self.assertEqual(0.1, round(algo.tokenizer.model.dropout, 4))
        self.assertEqual(1234, self.trainer.vocab_size)

    @patch('slangmod.ml.tokenizers.algo.Tokenizer.from_file')
    def test_from_file_called(self, mock):
        expected = 'file'
        mock.return_value = self.alternative
        algo = self.algo.from_file(expected)
        mock.assert_called_once_with(expected)
        self.assertIsNot(algo, self.algo)
        self.assertEqual(0.1, round(algo.tokenizer.model.dropout, 4))
        self.assertEqual(1234, self.trainer.vocab_size)

    @patch('slangmod.ml.tokenizers.algo.Tokenizer.from_pretrained')
    def test_from_pretrained_called_default(self, mock):
        expected = 'model'
        mock.return_value = self.alternative
        algo = self.algo.from_pretrained(expected)
        mock.assert_called_once_with(expected, 'main', None)
        self.assertIsNot(algo, self.algo)
        self.assertEqual(0.1, round(algo.tokenizer.model.dropout, 4))
        self.assertEqual(1234, self.trainer.vocab_size)

    @patch('slangmod.ml.tokenizers.algo.Tokenizer.from_pretrained')
    def test_from_pretrained_called_custom(self, mock):
        expected = 'model'
        mock.return_value = self.alternative
        algo = self.algo.from_pretrained(expected, 'branch', 'auth')
        mock.assert_called_once_with(expected, 'branch', 'auth')
        self.assertIsNot(algo, self.algo)
        self.assertEqual(0.1, round(algo.tokenizer.model.dropout, 4))
        self.assertEqual(1234, self.trainer.vocab_size)

    @patch('slangmod.ml.tokenizers.algo.Tokenizer.from_str')
    def test_from_str_called(self, mock):
        expected = 'JSON'
        mock.return_value = self.alternative
        algo = self.algo.from_str(expected)
        mock.assert_called_once_with(expected)
        self.assertIsNot(algo, self.algo)
        self.assertEqual(0.1, round(algo.tokenizer.model.dropout, 4))
        self.assertEqual(1234, self.trainer.vocab_size)

    @patch('slangmod.ml.tokenizers.algo.Tokenizer.train')
    def test_train_called(self, mock):
        expected = ['files']
        algo = self.algo.train(expected)
        mock.assert_called_once_with(expected, self.algo.trainer)
        self.assertIs(algo, self.algo)

    @patch('slangmod.ml.tokenizers.algo.Tokenizer.train_from_iterator')
    def test_train_from_iterator_called(self, mock):
        expected = ['files']
        algo = self.algo.train_from_iterator(expected)
        mock.assert_called_once_with(expected, self.algo.trainer)
        self.assertIs(algo, self.algo)

    def test_terminate_empty(self):
        actual = self.algo.terminate([])
        self.assertListEqual([self.special.eos_id], actual)

    def test_terminate_terminated(self):
        actual = self.algo.terminate([self.special.eos_id])
        self.assertListEqual([self.special.eos_id], actual)

    def test_terminate_unterminated(self):
        actual = self.algo.terminate([1, 2, 3, 4])
        self.assertListEqual([1, 2, 3, 4, self.special.eos_id], actual)

    def test_call_with_empty_batch(self):
        expected = []
        batch = []
        actual = self.algo(batch)
        self.assertListEqual(actual, expected)

    def test_call_with_empty_str(self):
        expected = [[self.special.eos_id]]
        batch = ['']
        actual = self.algo(batch)
        self.assertListEqual(actual, expected)

    def test_call_with_unterminated_list(self):
        expected = [[1, 0, self.special.eos_id]]
        batch = ['[UNK][PAD]']
        actual = self.algo(batch)
        self.assertListEqual(actual, expected)

    def test_call_with_terminated_list(self):
        expected = [[1, 0, self.special.eos_id]]
        batch = ['[UNK][PAD][EOS]']
        actual = self.algo(batch)
        self.assertListEqual(actual, expected)

    def test_call_with_tuple(self):
        expected = [[1, 0, self.special.eos_id]]
        batch = '[UNK][PAD]',
        actual = self.algo(batch)
        self.assertListEqual(actual, expected)

    def test_call_with_series(self):
        expected = [[1, 0, self.special.eos_id]]
        batch = Series(['[UNK][PAD]'])
        actual = self.algo(batch)
        self.assertListEqual(actual, expected)

    @patch('slangmod.ml.tokenizers.algo.Tokenizer.get_vocab')
    def test_access_tokenizer_attributes(self, mock):
        actual = self.algo.get_vocab
        self.assertIs(actual, mock)


class TestMisc(unittest.TestCase):

    def setUp(self):
        self.pad = AddedToken('[PAD]', special=True)
        self.unk = AddedToken('[UNK]', special=True)
        self.eos = AddedToken('[EOS]', special=True)
        self.cls = AddedToken('[CLS]', special=True)
        self.mask = AddedToken('[MASK]', special=True)
        self.special = Special(
            [self.cls],
            self.mask,
            pad=self.pad,
            unk=self.unk,
            eos=self.eos
        )
        self.model = BPE(unk_token=self.unk.content)
        self.tokenizer = Tokenizer(self.model)
        self.algo = Algo(self.special, self.tokenizer)

    def test_repr(self):
        self.assertEqual('BPE(...)', repr(self.algo))

    def test_pickle_works(self):
        _ = pickle.loads(pickle.dumps(self.algo))


if __name__ == '__main__':
    unittest.main()
