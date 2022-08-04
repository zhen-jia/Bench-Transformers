"""Register models that are converted from Huggingface transformers."""
# pylint: disable=protected-access, missing-function-docstring
import raf
from raf.frontend import from_pytorch

from raf._ffi.pass_ import InferType, ExprAppend, ExtractBinding
from raf._core.module import IRModule
from raf._core.ndarray import Symbol
from raf.frontend.model import FrameworkModel
from raf.testing import one_hot_torch, randn_torch

from tvm import relay

from raf_bencher import RAFBencher
from logger import get_logger
from registry import reg_model, get_model_bencher

from modeling_gpt_neo import GPTNeoForCausalLM
from modeling_gpt2 import GPT2LMHeadModel

from dataclasses import dataclass, field

from typing import Optional
from raf import distributed as dist

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoTokenizer,
    HfArgumentParser,
    AdamW,
    get_scheduler,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)

import numpy as np
logger = get_logger("RAF-TorchNLP")  # pylint: disable=invalid-name


def get_raf_func_output_var(func):
    """A helper function to get the output Var of the given function."""
    body = func.body
    while not isinstance(body, relay.Var):
        if isinstance(body, relay.Let):
            body = body.body
        else:
            raise NotImplementedError("Not supported type: ", type(body))
    return body


def transformer_common(model_name, batch_size, seq_length, dtype, include_orig_model):
    """The utility of processing transformer models.

    Parameters
    ----------
    model_name: str
        The model name.

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

    config = CONFIG_MAPPING[model_name]()
    num_layers = 24
    hidden_size = 2048
    device = "cuda" 
    
    def customize_model_config(config):
        config.vocab_size = config.vocab_size + (16 - config.vocab_size % 16) % 16
        if model_name == "gpt_neo":
            config.num_layers = num_layers
            config.hidden_size = hidden_size
            config.attention_layers = (['global'] + ['local']) * num_layers
        if model_name == "gpt2":
            config.n_layer = num_layers
            config.n_embd = hidden_size
            config.n_head = 16
        return config
    config = customize_model_config(config)
    if model_name == "gpt_neo":
        py_model = GPTNeoForCausalLM(config)
    elif model_name == "gpt2":
        py_model = GPT2LMHeadModel(config)


    t_model = py_model
    for n, p in t_model.named_parameters():
        print(" name is ", n , "shape is ", p.shape)

'''
    input_shape = [batch_size, seq_length]
    np_x = np.random.randint(0, 10000, input_shape) 
    #t_x = torch.tensor(np_x)
    t_model.eval()
    if dtype == "float16":
        t_model.half()
    
    t_dy = randn_torch((), std=0.0, mean=1.0, requires_grad=False, dtype=dtype) 


    #reshape_output = ref_bencher.kwargs.get("reshape_output", None)
    #seq_length = input_shape[1]
    mask = np.ones(input_shape)
    m_x = raf.ndarray(np_x, device=device) 
    m_mask = raf.ndarray(mask, device=device)
    try:
        #m_x = raf.ndarray(ref_bencher.args[0].numpy())
        inputs_shape = {
                "input_ids": (input_shape, "int64"),
                "attention_mask": (input_shape, "int64"),
                #  "labels": (input_shape, "int64"),
                }
        m_model = from_pytorch(t_model, inputs_shape)
        #print("m_mode is ", m_model)
        record = m_model._internal(m_x, m_mask)
        mod = record.mod
        mod = InferType()(mod)
        #print("mod is \n", mod)
        func = mod["main"]
        ret_var = get_raf_func_output_var(func)
        if isinstance(ret_var.checked_type, relay.TupleType):
            ret = Symbol.from_expr(ret_var)
            ret = ret[0]
            ret = ExtractBinding(ret._Symbol__handle, [ret_var])
            new_body = ExprAppend(func.body, ret)
            new_func = relay.Function(func.params, new_body)
            new_mod = IRModule.from_expr(new_func)
            m_model = FrameworkModel(
                new_mod,
                new_mod,
                m_model._FrameworkModel__arg_params,
                m_model._FrameworkModel__aux_params,
            )
    except Exception as err:  # pylint: disable=broad-except
        raise RuntimeError("Failed to convert model to RAF: %s" % (str(err)))

    m_dy, _ = randn_torch((),std=0.0, mean=1.0, requires_grad=False, dtype=dtype, device=device) 
    #raf.array(ref_bencher.dy.numpy())
    m_ytrue, _ = one_hot_torch(input_shape, 1000, device=device) 

    del t_model 
    #if not include_orig_model:
    #    del ref_bencher.model
    #    del ref_bencher.args
    #    del ref_bencher.dy
    #    del ref_bencher.y_true
    #    ref_bencher = None
    bencher = RAFBencher(
        m_model,
        input_shape,
        [m_x, m_mask],
        m_dy,
        m_ytrue,
        #ref_bencher=ref_bencher,
        #reshape_output=reshape_output,
    )
    #t = bencher.bench(device="cuda", warmup=50, number=50, train=True, optimizer="LANS")
    t = bencher.bench(device="cuda", warmup=50, number=50, train=True, data_parallel=True, zero_opt=2, optimizer="LANS")

    print("t is ", t)
    dist.RemoveCommunicator()


#@reg_model("raf")
#def gpt2(batch_size, seq_length, dtype, include_orig_model):
#    return transformer_common("gpt2", batch_size, seq_length, dtype, include_orig_model)

'''
if __name__ == "__main__":
    transformer_common("gpt_neo", 16, 512, "float16", False)

