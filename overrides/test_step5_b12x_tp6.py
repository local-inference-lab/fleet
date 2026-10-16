import pickle
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.models.step5.nvidia.model import (
    _validate_step5_dsa_parallel_geometry,
    pad_step5_attention_weight,
)


ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
HIDDEN, HEAD_DIM, PROXY_DIM = 64, 8, 4
SOURCE_HEADS, SOURCE_GROUPS, SOURCE_INDEX_HEADS = 64, 4, 16
HEADS, GROUPS, INDEX_HEADS, TP = 96, 12, 48, 6


def config():
    return SimpleNamespace(
        num_attention_heads=HEADS,
        num_attention_groups=GROUPS,
        attention_other_setting={
            "attention_type": "sliding_attention",
            "num_attention_heads": HEADS,
            "num_attention_groups": GROUPS,
            "head_dim": HEAD_DIM,
        },
        sparse_config={
            "enabled": True,
            # Validated by the Step 5 config; the padding itself never reads it.
            "proxy_dim": 256,
            "sparse_indexer_num_heads": INDEX_HEADS,
            "sparse_indexer_num_k_heads": 1,
            "num_provider_groups": GROUPS,
        },
        head_padding={
            "num_attention_heads": SOURCE_HEADS,
            "num_attention_groups": SOURCE_GROUPS,
            "sparse_indexer_num_heads": SOURCE_INDEX_HEADS,
        },
    )


def gated_attention(x, wq, wk, wv, wg, wo, heads, groups):
    tokens = x.shape[0]
    q = (x @ wq.T).view(tokens, heads, HEAD_DIM)
    k = (x @ wk.T).view(tokens, groups, HEAD_DIM)
    v = (x @ wv.T).view(tokens, groups, HEAD_DIM)
    k = k.repeat_interleave(heads // groups, dim=1)
    v = v.repeat_interleave(heads // groups, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) * HEAD_DIM**-0.5
    mask = torch.ones(tokens, tokens, dtype=torch.bool).tril()
    out = torch.einsum("hqk,khd->qhd", scores.masked_fill(~mask, -torch.inf).softmax(-1), v)
    out = out * torch.sigmoid(x @ wg.T)[..., None]
    return out.flatten(1) @ wo.T


def load(name, tensor):
    return pad_step5_attention_weight(config(), f"model.layers.3.self_attn.{name}", tensor)


class Step5PaddingTest(unittest.TestCase):
    def test_source_patches_are_present(self):
        step5 = ROOT / "models/step5/nvidia"
        self.assertIn("DeviceCapability(12, 0)", (step5 / "sparse_attention.py").read_text())
        self.assertIn("in ((9, 0), (12, 0))", (step5 / "flashinfer_sparse.py").read_text())
        self.assertIn("pad_step5_attention_weight", (step5 / "mtp.py").read_text())
        self.assertIn('"VLLM_STEP5_MOE_REDUCE_BF16"', (step5 / "model.py").read_text())
        b12x = (ROOT / "model_executor/layers/fused_moe/b12x.py").read_text()
        self.assertIn("MoEActivation.SWIGLUSTEP:\n            limit = 7.0", b12x)
        oracle = (ROOT / "model_executor/layers/fused_moe/oracle/unquantized.py").read_text()
        self.assertIn('"humming", "b12x", "flashinfer_b12x"', oracle)
        ipc = (ROOT / "distributed/device_communicators/flashinfer_pcie_ipc_all_reduce.py").read_text()
        self.assertIn("inp.shape[0] <= _MAX_TOKENS", ipc)
        tuner = (ROOT.parent / "flashinfer/autotuner/autotuner.py").read_text()
        self.assertIn('"FLASHINFER_AUTOTUNE_GPU_NAME"', tuner)
        gate = (ROOT / "model_executor/layers/fused_moe/router/gate_linear.py").read_text()
        self.assertIn("current_platform.is_device_capability((12, 0))", gate)

    def test_padded_geometry_fits_tp6_dsa_kernels(self):
        _validate_step5_dsa_parallel_geometry(
            total_num_heads=HEADS,
            total_num_kv_heads=GROUPS,
            indexer_num_heads=INDEX_HEADS,
            index_tp_size=GROUPS,
            tp_size=TP,
        )

    def test_tp6_shards_reproduce_source_attention(self):
        torch.manual_seed(0)
        x = torch.randn(7, HIDDEN, dtype=torch.float64)
        weights = {
            "q_proj.weight": torch.randn(SOURCE_HEADS * HEAD_DIM, HIDDEN, dtype=torch.float64),
            "k_proj.weight": torch.randn(SOURCE_GROUPS * HEAD_DIM, HIDDEN, dtype=torch.float64),
            "v_proj.weight": torch.randn(SOURCE_GROUPS * HEAD_DIM, HIDDEN, dtype=torch.float64),
            "g_proj.weight": torch.randn(SOURCE_HEADS, HIDDEN, dtype=torch.float64),
            "o_proj.weight": torch.randn(HIDDEN, SOURCE_HEADS * HEAD_DIM, dtype=torch.float64),
        }
        expected = gated_attention(
            x, *weights.values(), heads=SOURCE_HEADS, groups=SOURCE_GROUPS
        )
        padded = {name: load(name, tensor) for name, tensor in weights.items()}
        self.assertEqual(padded["q_proj.weight"].shape, (HEADS * HEAD_DIM, HIDDEN))
        self.assertEqual(padded["k_proj.weight"].shape, (GROUPS * HEAD_DIM, HIDDEN))
        self.assertEqual(padded["o_proj.weight"].shape, (HIDDEN, HEADS * HEAD_DIM))

        local_heads, local_groups = HEADS // TP, GROUPS // TP
        actual = torch.zeros_like(expected)
        for rank in range(TP):
            heads = slice(rank * local_heads, (rank + 1) * local_heads)
            groups = slice(rank * local_groups, (rank + 1) * local_groups)
            rows = slice(heads.start * HEAD_DIM, heads.stop * HEAD_DIM)
            kv_rows = slice(groups.start * HEAD_DIM, groups.stop * HEAD_DIM)
            actual += gated_attention(
                x,
                padded["q_proj.weight"][rows],
                padded["k_proj.weight"][kv_rows],
                padded["v_proj.weight"][kv_rows],
                padded["g_proj.weight"][heads],
                padded["o_proj.weight"][:, rows],
                heads=local_heads,
                groups=local_groups,
            )
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_replicated_provider_groups_score_like_their_source(self):
        torch.manual_seed(1)
        x = torch.randn(5, HIDDEN)
        key = torch.randn(9, PROXY_DIM)
        wq = torch.randn(SOURCE_INDEX_HEADS * PROXY_DIM, HIDDEN)
        ww = torch.randn(SOURCE_INDEX_HEADS, HIDDEN)

        def scores(q_weight, w_weight, groups):
            q = (x @ q_weight.T).view(x.shape[0], groups, -1, PROXY_DIM)
            w = (x @ w_weight.T).view(x.shape[0], groups, -1)
            return torch.einsum("tghd,kd->tghk", q, key).relu().mul(w[..., None]).sum(2)

        expected = scores(wq, ww, SOURCE_GROUPS)
        actual = scores(
            load("sparse_indexer_q.weight", wq), load("sparse_indexer_w.weight", ww), GROUPS
        )
        replicas = GROUPS // SOURCE_GROUPS
        torch.testing.assert_close(actual, expected.repeat_interleave(replicas, dim=1))

    def test_ssmax_and_unpadded_weights(self):
        ssmax = torch.arange(SOURCE_HEADS, dtype=torch.float32)
        self.assertEqual(load("ssmax_s", ssmax).shape, (HEADS,))
        norm = torch.randn(HEAD_DIM)
        self.assertIs(load("q_norm.weight", norm), norm)
        index_k = torch.randn(PROXY_DIM, HIDDEN)
        self.assertIs(load("sparse_indexer_k.weight", index_k), index_k)
        vision = torch.randn(3 * HIDDEN, HIDDEN)
        self.assertIs(
            pad_step5_attention_weight(
                config(), "vision_model.transformer.resblocks.0.attn.qkv_proj.weight", vision
            ),
            vision,
        )

    def test_mtp_draft_inherits_padded_geometry(self):
        from transformers import PretrainedConfig

        from vllm.config.speculative import SpeculativeConfig

        text = PretrainedConfig(
            num_attention_heads=SOURCE_HEADS,
            num_attention_groups=SOURCE_GROUPS,
            num_nextn_predict_layers=3,
            attention_other_setting={
                "attention_type": "sliding_attention",
                "num_attention_heads": SOURCE_HEADS,
                "num_attention_groups": SOURCE_GROUPS,
                "head_dim": 192,
            },
            sparse_config={
                "proxy_dim": 256,
                "topk": 512,
                "sparse_indexer_num_heads": SOURCE_INDEX_HEADS,
                "num_provider_groups": SOURCE_GROUPS,
            },
        )
        text.model_type = "step5"
        target = PretrainedConfig(
            text_config=text, architectures=["Step5ForConditionalGeneration"]
        )
        target.model_type = "step5_vl"
        served = config()
        overrides = {
            "text_config": {
                "num_attention_heads": HEADS,
                "num_attention_groups": GROUPS,
                "attention_other_setting": {
                    "num_attention_heads": HEADS,
                    "num_attention_groups": GROUPS,
                },
                "sparse_config": {
                    "sparse_indexer_num_heads": INDEX_HEADS,
                    "num_provider_groups": GROUPS,
                },
                "head_padding": served.head_padding,
            }
        }
        compose = SpeculativeConfig.compose_draft_hf_overrides(overrides)
        pickle.dumps(compose)
        draft = compose(target)
        self.assertEqual(draft.architectures, ["Step5MTP"])
        self.assertEqual(draft.num_attention_heads, HEADS)
        self.assertEqual(draft.num_attention_groups, GROUPS)
        self.assertEqual(draft.attention_other_setting["num_attention_heads"], HEADS)
        self.assertEqual(draft.attention_other_setting["head_dim"], 192)
        self.assertEqual(draft.sparse_config["num_provider_groups"], GROUPS)
        self.assertEqual(draft.sparse_config["topk"], 512)
        self.assertEqual(draft.head_padding, served.head_padding)
        self.assertIs(
            SpeculativeConfig.compose_draft_hf_overrides({"max_model_len": 1}),
            SpeculativeConfig.hf_config_override,
        )

    def test_tp6_vocab_padding_shards_step5_vocabulary(self):
        from unittest import mock

        import vllm.model_executor.layers.vocab_parallel_embedding as vocab

        def embedding(tp_size):
            with (
                mock.patch.object(vocab, "get_tensor_model_parallel_rank", return_value=0),
                mock.patch.object(
                    vocab, "get_tensor_model_parallel_world_size", return_value=tp_size
                ),
            ):
                return vocab.VocabParallelEmbedding(128896, 8)

        self.assertEqual(embedding(6).num_embeddings_per_partition, 21504)
        self.assertEqual(embedding(4).num_embeddings_per_partition, 128896 // 4)
        self.assertEqual(embedding(8).num_embeddings_per_partition, 128896 // 8)

    def test_unpadded_config_is_unchanged(self):
        plain = SimpleNamespace(num_attention_heads=SOURCE_HEADS, num_attention_groups=SOURCE_GROUPS)
        q = torch.randn(SOURCE_HEADS * HEAD_DIM, HIDDEN)
        self.assertIs(
            pad_step5_attention_weight(plain, "model.layers.3.self_attn.q_proj.weight", q), q
        )


if __name__ == "__main__":
    unittest.main()
