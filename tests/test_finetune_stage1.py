from __future__ import annotations

import sys
import types
import importlib
from pathlib import Path

import pytest


class _NoGrad:
    def __call__(self, fn=None):
        if fn is None:
            return self
        return fn

    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


class _AdamW:
    def __init__(self, params, lr=None, weight_decay=None):
        self.param_groups = params if isinstance(params, list) else [
            {"params": list(params), "lr": lr, "weight_decay": weight_decay}
        ]


torch_stub = types.ModuleType("torch")
torch_stub.no_grad = lambda: _NoGrad()
torch_stub.optim = types.SimpleNamespace(AdamW=_AdamW)
torch_stub.Tensor = object

pytorch_lightning_stub = types.ModuleType("pytorch_lightning")
pytorch_lightning_stub.LightningModule = object

model_finetune_stub = types.ModuleType("stack.model_finetune")
model_finetune_stub.ICL_FinetunedModel = object
finetune_datamodule_stub = types.ModuleType("stack.finetune.datamodule")
finetune_utils_stub = types.ModuleType("stack.finetune.utils")
finetune_datasets_stub = types.ModuleType("stack.finetune.datasets")


class FakeParam:
    def __init__(self, name):
        self.name = name
        self.requires_grad = True

    def numel(self):
        return 1


class FakeModule:
    def __init__(self, prefix, count=2):
        self._params = [FakeParam(f"{prefix}.{idx}") for idx in range(count)]

    def parameters(self):
        return iter(self._params)


class FakeLinear:
    def __init__(self, prefix):
        self.weight = FakeParam(f"{prefix}.weight")
        self.bias = FakeParam(f"{prefix}.bias")

    def parameters(self):
        return iter([self.weight, self.bias])


class FakeOutputMlp:
    def __init__(self):
        self._layers = [
            FakeLinear("output_mlp.0"),
            object(),
            object(),
            FakeLinear("output_mlp.3"),
        ]

    def __getitem__(self, index):
        return self._layers[index]

    def parameters(self):
        for layer in self._layers:
            if hasattr(layer, "parameters"):
                yield from layer.parameters()


class FakeModel:
    def __init__(self):
        self.query_pos_embedding = FakeParam("query_pos_embedding")
        self.gene_pos_embedding = FakeParam("gene_pos_embedding")
        self.gene_reduction = FakeModule("gene_reduction")
        self.layers = [FakeModule("layers.0")]
        self.cls = FakeModule("cls")
        self.output_mlp = FakeOutputMlp()

    def named_parameters(self):
        yield "query_pos_embedding", self.query_pos_embedding
        yield "gene_pos_embedding", self.gene_pos_embedding
        for idx, param in enumerate(self.gene_reduction.parameters()):
            yield f"gene_reduction.{idx}", param
        for layer_idx, layer in enumerate(self.layers):
            for param_idx, param in enumerate(layer.parameters()):
                yield f"layers.{layer_idx}.{param_idx}", param
        for idx, param in enumerate(self.cls.parameters()):
            yield f"cls.{idx}", param
        yield "output_mlp.0.weight", self.output_mlp[0].weight
        yield "output_mlp.0.bias", self.output_mlp[0].bias
        yield "output_mlp.3.weight", self.output_mlp[3].weight
        yield "output_mlp.3.bias", self.output_mlp[3].bias

    def parameters(self):
        for _, param in self.named_parameters():
            yield param


@pytest.fixture()
def lightning_cls(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "pytorch_lightning", pytorch_lightning_stub)
    monkeypatch.setitem(sys.modules, "stack.model_finetune", model_finetune_stub)
    monkeypatch.setitem(sys.modules, "stack.finetune.datamodule", finetune_datamodule_stub)
    monkeypatch.setitem(sys.modules, "stack.finetune.utils", finetune_utils_stub)
    monkeypatch.setitem(sys.modules, "stack.finetune.datasets", finetune_datasets_stub)
    sys.modules.pop("stack.finetune.lightning", None)

    module = importlib.import_module("stack.finetune.lightning")
    yield module.LightningFinetunedModel

    sys.modules.pop("stack.finetune.lightning", None)


def make_lightning_model(lightning_cls):
    lightning_model = object.__new__(lightning_cls)
    lightning_model.model = FakeModel()
    lightning_model.teacher_model = FakeModel()
    lightning_model.finetune_strategy = "stage1"
    lightning_model.learning_rate = 1e-5
    lightning_model.head_lr = 1e-4
    lightning_model.decoder_lr = 1e-5
    lightning_model.weight_decay = 0.003
    lightning_model.scheduler_config = {}
    return lightning_model


def test_stage1_freezes_backbone_and_trains_query_cls_decoder_bias(lightning_cls):
    lightning_model = make_lightning_model(lightning_cls)

    lightning_model._apply_finetune_strategy()

    trainable = {
        name
        for name, param in lightning_model.model.named_parameters()
        if param.requires_grad
    }
    assert trainable == {
        "query_pos_embedding",
        "cls.0",
        "cls.1",
        "output_mlp.3.bias",
    }


def test_stage1_optimizer_uses_head_and_decoder_learning_rates(lightning_cls):
    lightning_model = make_lightning_model(lightning_cls)
    lightning_model._apply_finetune_strategy()

    config = lightning_model.configure_optimizers()

    groups = config["optimizer"].param_groups
    assert [group["lr"] for group in groups] == [1e-4, 1e-4, 1e-5]
    assert groups[0]["weight_decay"] == 0.0
    assert groups[1]["weight_decay"] == 0.003
    assert groups[2]["weight_decay"] == 0.0
    assert groups[2]["params"] == [lightning_model.model.output_mlp[3].bias]


def test_student_only_checkpoint_restores_teacher_from_student(lightning_cls):
    lightning_model = make_lightning_model(lightning_cls)
    model_value = object()
    state_dict = {
        "model.query_pos_embedding": model_value,
        "model.output_mlp.3.bias": object(),
    }
    checkpoint = {"state_dict": state_dict}

    lightning_model.on_load_checkpoint(checkpoint)

    assert checkpoint["state_dict"]["teacher_model.query_pos_embedding"] is model_value
    assert "teacher_model.output_mlp.3.bias" in checkpoint["state_dict"]
