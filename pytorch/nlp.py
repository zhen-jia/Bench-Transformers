"""Register NLP models from PyTorch."""
# pylint: disable=not-callable, missing-function-docstring, protected-access, unused-argument
import os

import numpy as np
import torch
import transformers

from ..logger import get_logger
from ..registry import reg_model
from .torch_bencher import TorchBencher
from .utils import randn_torch, one_hot_torch

logger = get_logger("PyTorch-NLP")  # pylint: disable=invalid-name


class ConvertNLPContext:
    """The context to deal with TOKENIZERS_PARALLELISM."""

    def __init__(self):
        self.tokenizers_parallelism = None

    def __enter__(self):
        if "TOKENIZERS_PARALLELISM" in os.environ:
            self.tokenizers_parallelism = os.environ["TOKENIZERS_PARALLELISM"]
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def __exit__(self, ptype, value, trace):
        if self.tokenizers_parallelism is not None:
            os.environ["TOKENIZERS_PARALLELISM"] = self.tokenizers_parallelism
        else:
            del os.environ["TOKENIZERS_PARALLELISM"]


def transformer_common(model_or_config, batch_size, seq_length, dtype, include_orig_model):
    """The utility of processing transformer models.

    Parameters
    ----------
    model_or_config: Union[Dict[str, Any], torch.nn.Module]
        The Huggingface PyTorch model configuration, or the PyTorch model.

    batch_size: int
        Batch size.

    seq_length: Optional[int]
        The sequence length. If None, 128 will be used.

    dtype: str
        The data type. Default is float32.

    include_orig_model: bool
        Whether to include the original model as the reference.

    Returns
    -------
    mod_n_shape: Tuple[raf.Model, Tuple[int, int]]
        The converted model and input shape.
    """
    if isinstance(model_or_config, transformers.configuration_utils.PretrainedConfig):
        assert hasattr(model_or_config, "architectures"), '"architectures" is missing in the config'
        model_cls = model_or_config.architectures[0]
        model_or_config.use_cache = False  # Disable model cache to avoid unnecessary model outputs.
        assert hasattr(transformers, model_cls), "%s is not supported in transformers" % model_cls
        t_model = getattr(transformers, model_cls)(model_or_config)
    else:
        assert isinstance(model_or_config, torch.nn.Module), "The model should be a PyTorch model"
        model_cls = model_or_config.__class__.__name__
        t_model = model_or_config

    torch.manual_seed(0)  # Fix the random seed
    seq_length = seq_length if seq_length is not None else 128
    input_shape = [batch_size, seq_length]

    np_x = np.random.randint(0, 10000, input_shape)
    t_x = torch.tensor(np_x)
    t_model.eval()
    if dtype == "float16":
        t_model.half()
    t_dy = randn_torch((), std=0.0, mean=1.0, requires_grad=False, dtype=dtype)

    if model_cls.find("ForSequenceClassification") != -1:
        t_y = t_model(t_x)
        t_ytrue = one_hot_torch(size=batch_size, num_classes=t_y[0].shape[1])
        output_shape = None
    elif model_cls.find("LM") != -1 or model_cls.find("LN") != -1 or model_cls.find("Bart") != -1:
        # Language model (e.g., BertForMaskedLM, GPT2LMHeadModel, BertForPreTrainingPreLN)
        if hasattr(t_model, "bert_config"):
            vocab_size = t_model.bert_config.vocab_size
        elif hasattr(t_model, "config"):
            vocab_size = t_model.config.vocab_size
        else:
            raise RuntimeError("Do not know how to get model config for %s" % type(t_model))
        t_ytrue = one_hot_torch(size=(batch_size, seq_length), num_classes=vocab_size)
        output_shape = (batch_size * seq_length, vocab_size)
    else:
        raise ValueError("Unsupported model type: %s" % model_cls)

    bencher = TorchBencher(t_model, input_shape, [t_x], t_dy, t_ytrue, reshape_output=output_shape)
    torch.cuda.empty_cache()
    return bencher


@reg_model("torch")
def bert_base_classify(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("bert-base-uncased")
        config.architectures = ["BertForSequenceClassification"]
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)


@reg_model("torch")
def bert_test_classify(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("bert-base-uncased")
        config.num_hidden_layers = 1
        config.architectures = ["BertForSequenceClassification"]
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)


@reg_model("torch")
def bert_large_classify(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("bert-large-uncased")
        config.architectures = ["BertForSequenceClassification"]
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)


@reg_model("torch")
def gpt2_classify(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.GPT2Config()
        config.pad_token_id = -1
        config.architectures = ["GPT2ForSequenceClassification"]
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)


@reg_model("torch")
def bert_base_mlm(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("bert-base-uncased")
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)


@reg_model("torch")
def bert_test_mlm(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("bert-base-uncased")
        config.num_hidden_layers = 1
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)


@reg_model("torch")
def bert_large_mlm(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("bert-large-uncased")
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)


@reg_model("torch")
def gpt2(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("gpt2")
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)


@reg_model("torch")
def gpt2_large(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("gpt2-large")
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)


@reg_model("torch")
def gpt2_test(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("gpt2")
        config.n_layer = 1
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)


@reg_model("torch")
def bart_base(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("facebook/bart-base")
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)


@reg_model("torch")
def roberta_base(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("roberta-base")
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)

@reg_model("torch")
def roberta_large(batch_size, seq_length, dtype, include_orig_model):
    with ConvertNLPContext():
        config = transformers.AutoConfig.from_pretrained("roberta-large")
        return transformer_common(config, batch_size, seq_length, dtype, include_orig_model)
