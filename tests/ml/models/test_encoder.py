import unittest
from unittest.mock import patch
import torch as pt
from swak.pt.misc import Identity
from swak.pt.blocks import ActivatedBlock

from slangmod.ml.models import (
    SelfAttention,
    EncoderLayer,
    Encoder,
    Sinusoidal
)


class TestDefaultAttributes(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.context = 32
        self.vocab = 128
        self.attention = SelfAttention(self.mod_dim, self.n_heads)
        self.pos_enc = Sinusoidal(self.mod_dim, self.context)
        self.feedforward = ActivatedBlock(self.mod_dim)
        self.layer = EncoderLayer(
            self.attention,
            self.feedforward,
            self.pos_enc
        )
        self.encode = Encoder(self.vocab, self.layer)

    def test_has_vocab(self):
        self.assertTrue(hasattr(self.encode, 'vocab'))

    def test_vocab(self):
        self.assertIsInstance(self.encode.vocab, int)
        self.assertEqual(self.vocab, self.encode.vocab)

    def test_has_n_layers(self):
        self.assertTrue(hasattr(self.encode, 'n_layers'))

    def test_n_layers(self):
        self.assertIsInstance(self.encode.n_layers, int)
        self.assertEqual(2, self.encode.n_layers)

    def test_has_pad_id(self):
        self.assertTrue(hasattr(self.encode, 'pad_id'))

    def test_pad_id(self):
        self.assertIsInstance(self.encode.pad_id, int)
        self.assertEqual(0, self.encode.pad_id)

    def test_has_pos_enc(self):
        self.assertTrue(hasattr(self.encode, 'pos_enc'))

    def test_pos_enc(self):
        self.assertIsInstance(self.encode.pos_enc, Identity)

    def test_has_bias(self):
        self.assertTrue(hasattr(self.encode, 'bias'))

    def test_bias(self):
        self.assertIsInstance(self.encode.bias, bool)
        self.assertTrue(self.encode.bias)

    def test_has_dropout(self):
        self.assertTrue(hasattr(self.encode, 'dropout'))

    def test_dropout(self):
        self.assertEqual(0.1, self.encode.dropout)

    def test_has_scale_grad_by_freq(self):
        self.assertTrue(hasattr(self.encode, 'scale_grad_by_freq'))

    def test_scale_grad_by_freq(self):
        self.assertTrue(self.encode.scale_grad_by_freq)

    def test_has_device(self):
        self.assertTrue(hasattr(self.encode, 'device'))

    def test_device(self):
        self.assertIsInstance(self.encode.device, pt.device)
        self.assertEqual('cpu', self.encode.device.type)

    def test_has_dtype(self):
        self.assertTrue(hasattr(self.encode, 'dtype'))

    def test_dtype(self):
        self.assertIs(self.encode.dtype, pt.float)

    def test_has_layers(self):
        self.assertTrue(hasattr(self.encode, 'layers'))

    def test_layers(self):
        self.assertIsInstance(self.encode.layers, pt.nn.ModuleList)
        for layer in self.encode.layers:
            self.assertIsInstance(layer, EncoderLayer)
            self.assertIs(layer.dtype, self.encode.dtype)
            self.assertEqual(layer.device, self.encode.device)

    def test_has_embed(self):
        self.assertTrue(hasattr(self.encode, 'embed'))

    def test_embed(self):
        self.assertIsInstance(self.encode.embed, pt.nn.Embedding)
        self.assertTrue(self.encode.embed.scale_grad_by_freq)
        self.assertEqual(self.vocab, self.encode.embed.num_embeddings)
        self.assertEqual(self.mod_dim, self.encode.embed.embedding_dim)
        self.assertEqual(0, self.encode.embed.padding_idx)
        self.assertEqual(
            self.encode.scale_grad_by_freq,
            self.encode.embed.scale_grad_by_freq
        )
        self.assertIs(pt.float, self.encode.embed.weight.dtype)
        self.assertEqual('cpu', self.encode.embed.weight.device.type)

    def test_has_drop(self):
        self.assertTrue(hasattr(self.encode, 'drop'))

    def test_drop(self):
        self.assertIsInstance(self.encode.drop, pt.nn.Dropout)
        self.assertEqual(self.encode.dropout, self.encode.drop.p)

    def test_has_norm(self):
        self.assertTrue(hasattr(self.encode, 'norm'))

    def test_norm(self):
        self.assertIsInstance(self.encode.norm, pt.nn.LayerNorm)
        self.assertTupleEqual(
            (self.mod_dim,),
            self.encode.norm.normalized_shape
        )
        self.assertEqual(self.encode.norm.eps, self.encode.layers[0].norm1.eps)
        self.assertTrue(self.encode.norm.elementwise_affine)
        self.assertTupleEqual(
            (self.mod_dim,),
            self.encode.norm.bias.shape
        )
        self.assertEqual('cpu', self.encode.norm.weight.device.type)
        self.assertIs(pt.float, self.encode.norm.weight.dtype)

    def test_has_finalize(self):
        self.assertTrue(hasattr(self.encode, 'finalize'))

    def test_finalize(self):
        self.assertIsInstance(self.encode.finalize, pt.nn.Linear)
        self.assertEqual(self.mod_dim, self.encode.finalize.in_features)
        self.assertEqual(self.vocab, self.encode.finalize.out_features)
        self.assertEqual('cpu', self.encode.finalize.weight.device.type)
        self.assertIs(self.encode.finalize.weight.dtype, pt.float)
        self.assertTupleEqual(
            (self.vocab,),
            self.encode.finalize.bias.shape
        )

    def test_has_mod_dim(self):
        self.assertTrue(hasattr(self.encode, 'mod_dim'))

    def test_mod_dim(self):
        self.assertIsInstance(self.encode.mod_dim, int)
        self.assertEqual(self.mod_dim, self.encode.mod_dim)

    def test_has_context(self):
        self.assertTrue(hasattr(self.encode, 'context'))

    def test_context(self):
        self.assertIsInstance(self.encode.context, int)
        self.assertEqual(self.context, self.encode.context)

    def test_has_reset_parameters(self):
        self.assertTrue(hasattr(self.encode, 'reset_parameters'))

    def test_reset_parameters_callable(self):
        self.assertTrue(callable(self.encode.reset_parameters))

    def test_call_reset_parameters(self):
        layer0 = patch.object(self.encode.layers[0], 'reset_parameters')
        layer1 = patch.object(self.encode.layers[1], 'reset_parameters')
        embed = patch.object(self.encode.embed, 'reset_parameters')
        pos_enc = patch.object(self.encode.pos_enc, 'reset_parameters')
        finalize = patch.object(self.encode.finalize, 'reset_parameters')
        with layer0 as a, layer1 as b, embed as e, pos_enc as p, finalize as f:
            self.encode.reset_parameters()
            a.assert_called_once_with()
            b.assert_called_once_with()
            e.assert_called_once_with()
            p.assert_called_once_with()
            f.assert_called_once_with()


class TestAttributes(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.context = 32
        self.vocab = 128
        self.n_layers = 3
        self.pad_id = 1
        self.bias = False
        self.dropout = 0.2
        self.scale_grad_by_freq = False
        self.dtype = pt.double
        self.attention = SelfAttention(self.mod_dim, self.n_heads)
        self.pos_enc = Sinusoidal(self.mod_dim, self.context)
        self.feedforward = ActivatedBlock(self.mod_dim)
        self.layer = EncoderLayer(
            self.attention,
            self.feedforward,
            norm_cls='rms'
        )
        self.encode = Encoder(
            self.vocab,
            self.layer,
            self.n_layers,
            self.pad_id,
            self.pos_enc,
            self.bias,
            self.dropout,
            self.scale_grad_by_freq,
            dtype=self.dtype
        )

    def test_n_layers(self):
        self.assertEqual(self.n_layers, self.encode.n_layers)

    def test_pad_id(self):
        self.assertEqual(self.pad_id, self.encode.pad_id)

    def test_pos_enc(self):
        self.assertIsInstance(self.encode.pos_enc, Sinusoidal)

    def test_bias(self):
        self.assertEqual(self.bias, self.encode.bias)

    def test_dropout(self):
        self.assertEqual(self.dropout, self.encode.dropout)

    def test_scale_grad_by_freq(self):
        self.assertEqual(
            self.scale_grad_by_freq,
            self.encode.scale_grad_by_freq
        )

    def test_dtype(self):
        self.assertIs(self.dtype, self.encode.dtype)

    def test_layers(self):
        for layer in self.encode.layers:
            self.assertIs(layer.dtype, self.dtype)

    def test_embed(self):
        self.assertEqual(
            self.scale_grad_by_freq,
            self.encode.embed.scale_grad_by_freq
        )
        self.assertEqual(self.pad_id, self.encode.embed.padding_idx)
        self.assertEqual(
            self.encode.scale_grad_by_freq,
            self.encode.embed.scale_grad_by_freq
        )
        self.assertIs(self.encode.embed.weight.dtype, self.dtype)

    def test_norm_cls(self):
        self.assertIsInstance(self.encode.norm, pt.nn.RMSNorm)

    def test_norm(self):
        self.assertIs(self.dtype, self.encode.norm.weight.dtype)

    def test_finalize(self):
        self.assertIs(self.encode.finalize.weight.dtype, self.dtype)
        self.assertIsNone(self.encode.finalize.bias)

    def test_context(self):
        self.assertEqual(self.context, self.encode.context)

    def test_double_pos_enc_warns(self):
        layer = EncoderLayer(self.attention, self.feedforward, self.pos_enc)
        with self.assertWarns(UserWarning):
            _ = Encoder(self.vocab, layer, pos_enc=self.pos_enc)

    def test_no_pos_enc_raises(self):
        with self.assertRaises(TypeError):
            _ = Encoder(self.vocab, self.layer)

    def test_norm_identity_not_norm_first(self):
        el = EncoderLayer(self.attention, self.feedforward, norm_first=False)
        encode = Encoder(self.vocab, el, pos_enc=self.pos_enc)
        self.assertIsInstance(encode.norm, Identity)


class TestMasking(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.context = 32
        self.vocab = 128
        self.attention = SelfAttention(self.mod_dim, self.n_heads)
        self.pos_enc = Sinusoidal(self.mod_dim, self.context)
        self.feedforward = ActivatedBlock(self.mod_dim)
        self.layer = EncoderLayer(
            self.attention,
            self.feedforward,
            self.pos_enc
        )
        self.encode = Encoder(self.vocab, self.layer, 1)
        self.inp = pt.randint(
            0,
            self.vocab,
            (1, self.context),
            device='cpu',
            dtype=pt.long
        )
        self.out = pt.rand(1, self.context, self.mod_dim, device='cpu')
        self.attn_mask = pt.nn.Transformer.generate_square_subsequent_mask(
            self.context,
            device='cpu'
        )
        self.src_mask = pt.zeros(self.context, device='cpu')
        self.src_mask[1] = float('-inf')
        self.src_mask[4] = float('-inf')
        self.src_mask[7] = float('-inf')

    def test_default(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertIsNone(mask)
            self.assertIsInstance(is_causal, bool)
            self.assertTrue(is_causal)

    def test_is_causal_no_attn_mask_no_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, None, None, True)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertIsNone(mask)
            self.assertTrue(is_causal)

    def test_is_causal_attn_mask_no_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, self.attn_mask, None, True)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertIsNone(mask)
            self.assertTrue(is_causal)

    def test_is_causal_no_attn_mask_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, None, self.src_mask, True)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertIsNone(mask)
            self.assertTrue(is_causal)

    def test_is_causal_attn_mask_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, self.attn_mask, self.src_mask, True)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertIsNone(mask)
            self.assertTrue(is_causal)

    def test_is_not_causal_no_attn_mask_no_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, None, None, False)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertIsNone(mask)
            self.assertFalse(is_causal)

    def test_is_not_causal_attn_mask_no_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, self.attn_mask, None, False)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertIs(mask, self.attn_mask)
            self.assertFalse(is_causal)

    def test_is_not_causal_no_attn_mask_src_mask_unbatched(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, None, self.src_mask, False)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            pt.testing.assert_close(mask, expected)

    def test_is_not_causal_no_attn_mask_src_mask_batch_1(self):
        src_mask = self.src_mask.unsqueeze(0)
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, None, src_mask, False)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = src_mask.unsqueeze(0).expand(-1, self.context, -1)
            pt.testing.assert_close(mask, expected)

    def test_is_not_causal_no_attn_mask_src_mask_batched(self):
        src_mask = self.src_mask.unsqueeze(0).expand(3, -1)
        inp = self.inp.expand(3, -1)
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(inp, None, src_mask, False)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = src_mask.unsqueeze(-2).expand(-1, self.context, -1)
            pt.testing.assert_close(mask, expected)

    def test_is_not_causal_attn_mask_src_mask_unbatched(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, self.attn_mask, self.src_mask, False)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            pt.testing.assert_close(mask, expected + self.attn_mask)

    def test_is_not_causal_attn_mask_src_mask_batch_1(self):
        src_mask = self.src_mask.unsqueeze(0)
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(self.inp, self.attn_mask, src_mask, False)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = src_mask.unsqueeze(0).expand(-1, self.context, -1)
            pt.testing.assert_close(mask, expected + self.attn_mask)

    def test_is_not_causal_attn_mask_src_mask_batched(self):
        src_mask = self.src_mask.unsqueeze(0).expand(3, -1)
        inp = self.inp.expand(3, -1)
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(inp, self.attn_mask, src_mask, False)
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = src_mask.unsqueeze(-2).expand(-1, self.context, -1)
            pt.testing.assert_close(mask, expected + self.attn_mask)

    def test_2d_attn_mask_3d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask,
                self.src_mask.unsqueeze(0).unsqueeze(0),
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = expected.unsqueeze(0).unsqueeze(0) + self.attn_mask
            pt.testing.assert_close(mask, expected)

    def test_2d_attn_mask_4d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask,
                self.src_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = expected.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            pt.testing.assert_close(mask, expected + self.attn_mask)

    def test_3d_attn_mask_1d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask.unsqueeze(0),
                self.src_mask,
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = self.attn_mask.unsqueeze(0) + expected
            pt.testing.assert_close(mask, expected)

    def test_3d_attn_mask_2d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask.unsqueeze(0),
                self.src_mask.unsqueeze(0),
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = self.attn_mask.unsqueeze(0) + expected
            pt.testing.assert_close(mask, expected)

    def test_3d_attn_mask_3d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask.unsqueeze(0),
                self.src_mask.unsqueeze(0).unsqueeze(0),
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = self.attn_mask + expected.unsqueeze(0).unsqueeze(0)
            pt.testing.assert_close(mask, expected)

    def test_3d_attn_mask_4d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask.unsqueeze(0),
                self.src_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = (
                self.attn_mask.unsqueeze(0) +
                expected.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            pt.testing.assert_close(mask, expected)

    def test_4d_attn_mask_1d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask.unsqueeze(0).unsqueeze(0),
                self.src_mask,
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = self.attn_mask.unsqueeze(0).unsqueeze(0) + expected
            pt.testing.assert_close(mask, expected)

    def test_4d_attn_mask_2d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask.unsqueeze(0).unsqueeze(0),
                self.src_mask.unsqueeze(0),
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = (
                self.attn_mask.unsqueeze(0).unsqueeze(0) + expected)
            pt.testing.assert_close(mask, expected)

    def test_4d_attn_mask_3d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask.unsqueeze(0).unsqueeze(0),
                self.src_mask.unsqueeze(0).unsqueeze(0),
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = (
                self.attn_mask.unsqueeze(0).unsqueeze(0) +
                expected.unsqueeze(0)
            )
            pt.testing.assert_close(mask, expected)

    def test_4d_attn_mask_4d_src_mask(self):
        with patch.object(self.encode.layers[0], 'forward') as layer:
            layer.return_value = self.out
            _ = self.encode(
                self.inp,
                self.attn_mask.unsqueeze(0).unsqueeze(0),
                self.src_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                False
            )
            layer.assert_called_once()
            mask, is_causal = layer.call_args[0][1:]
            self.assertFalse(is_causal)
            expected = self.src_mask.unsqueeze(0).expand(self.context, -1)
            expected = (
                self.attn_mask.unsqueeze(0).unsqueeze(0) +
                expected.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            pt.testing.assert_close(mask, expected)


class TestUsage(unittest.TestCase):

    def setUp(self):
        self.mod_dim = 16
        self.n_heads = 2
        self.context = 32
        self.vocab = 128
        self.attention = SelfAttention(self.mod_dim, self.n_heads)
        self.pos_enc = Sinusoidal(self.mod_dim, self.context)
        self.feedforward = ActivatedBlock(self.mod_dim)
        self.layer = EncoderLayer(
            self.attention,
            self.feedforward
        )
        self.encode = Encoder(self.vocab, self.layer, 2, 0, self.pos_enc)
        self.inp = pt.randint(
            0,
            self.vocab,
            (1, self.context),
            device='cpu',
            dtype=pt.long
        )
        self.out = pt.rand(1, self.context, self.mod_dim, device='cpu')

    def test_embed_called(self):
        with patch.object(self.encode.embed, 'forward') as forward:
            forward.return_value = self.out
            _ = self.encode(self.inp)
            forward.assert_called_once_with(self.inp)

    def test_pos_enc_called(self):
        with patch.object(
            self.encode.embed,
            'forward',
            return_value = self.out
        ), patch.object(
            self.encode.pos_enc,
            'forward',
            return_value = self.out
        ) as forward:
            _ = self.encode(self.inp)
            forward.assert_called_once_with(self.out)

    def test_drop_called(self):
        with patch.object(
            self.encode.pos_enc,
            'forward',
            return_value = self.out
        ), patch.object(
            self.encode.drop,
            'forward',
            return_value = self.out
        ) as forward:
            _ = self.encode(self.inp)
            forward.assert_called_once_with(self.out)

    def test_2nd_layer_called(self):
        with patch.object(
            self.encode.layers[0],
            'forward',
            return_value = self.out
        ), patch.object(
            self.encode.layers[1],
            'forward',
            return_value = self.out
        ) as forward:
            _ = self.encode(self.inp)
            forward.assert_called_once_with(self.out, None, True)

    def test_norm_called(self):
        with patch.object(
            self.encode.layers[1],
            'forward',
            return_value = self.out
        ), patch.object(
            self.encode.norm,
            'forward',
            return_value = self.out
        ) as forward:
            _ = self.encode(self.inp)
            forward.assert_called_once_with(self.out)

    def test_finalize_called(self):
        with patch.object(
            self.encode.norm,
            'forward',
            return_value = self.out
        ), patch.object(
            self.encode.finalize,
            'forward',
            return_value = self.out
        ) as forward:
            _ = self.encode(self.inp)
            forward.assert_called_once_with(self.out)

    def test_1d(self):
        inp = pt.randint(
            0,
            self.vocab,
            (self.context,),
            device='cpu',
            dtype=pt.long
        )
        actual, = self.encode(inp)
        expected = 1, 128, 32
        self.assertTupleEqual(expected, actual.shape)

    def test_2d(self):
        inp = pt.randint(
            0,
            self.vocab,
            (7, self.context),
            device='cpu',
            dtype=pt.long
        )
        actual, = self.encode(inp)
        expected = 7, 128, 32
        self.assertTupleEqual(expected, actual.shape)


if __name__ == '__main__':
    unittest.main()
