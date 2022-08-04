"""
The module of wrapper classes to benchmark PyTorch models.
"""
# pylint: disable=too-many-arguments, too-many-statements
from importlib import import_module
import os
import timeit

import torch

from ..logger import get_logger
from ..model_bencher import ModelBencherBase
from ..utils import get_gpu_total_memory, get_tensor_size

logger = get_logger("Torch")  # pylint: disable=invalid-name


class DeviceModelSetter:
    """Utilities of setting up device and model.

    Parameters
    ----------
    device : str
        The target device.
    """

    def __init__(self, device):
        self.device = device

    def setup_model(self, model):
        """Setup the model."""
        model.to(self.device)
        model.train()
        return (model, model)

    def step(self):
        """A step after each iteration."""
        if self.device == "cuda":
            torch.cuda.synchronize()

    @property
    def autocast(self):
        """AMP context."""
        return torch.cuda.amp.autocast

    def optim(self, name, params, lr=0.1, **kwargs):  # pylint: disable=no-self-use
        """Wrap the optimizer."""
        if name == "SGD":
            momentum = kwargs.get("momentum", 0.01)
            return torch.optim.SGD(params, lr=lr, momentum=momentum)
        if name == "Adam":
            eps = kwargs.get("eps", 1e-8)
            return torch.optim.Adam(params, lr=lr, eps=eps)
        if name == "LANS":
            try:
                return import_module("apex.optimizers").FusedLANS(params)
            except ModuleNotFoundError:
                raise RuntimeError("apex.optimizers.FusedLANS is not imported corretly")
        raise ValueError("Unsupportde optimizer: " % name)

    @staticmethod
    def forward_amp_only():
        """Indicate whether the AMP should only cover the forward pass."""
        return True

    def finalize(self):
        """Finalize process."""

    def profile_latency_start(self):
        """Start profiling latency."""
        raise NotImplementedError

    def profile_latency_stop(self):
        """Stop profiling latency."""
        raise NotImplementedError

    def get_latency_stats(self):
        """Get and reset the latency stats."""
        raise NotImplementedError

    def profile_memory_start(self):
        """Start profiling memory."""

    def profile_memory_stop(self):
        """Stop profiling memory."""

    def reset_memory_stats(self):
        """Reset memory stats."""
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def max_memory_allocated(self, _):
        """Get max memory allocated in MBs."""
        if self.device == "cuda":
            return torch.cuda.max_memory_allocated() / 2**20
        raise RuntimeError(f"Memory profiler on {self.device} is not supported")

    def max_memory_reserved(self, _):
        """Get max memory reserved in MBs."""
        if self.device == "cuda":
            return torch.cuda.max_memory_reserved() / 2**20
        raise RuntimeError(f"Memory profiler on {self.device} is not supported")

    @staticmethod
    def create(options, device):
        """The factory to dispatch the target setter."""
        name = options.get("ltc", None)
        if name == "xla":
            return TorchXLASetter(options)
        if name == "razor":
            return RazorSetter(options)
        return DeviceModelSetter(device)


class TorchXLASetter(DeviceModelSetter):
    """The settings for LazyTensorCore with PyTorch/XLA. Note that we do not
    support mmeory profiling for PT-XLA yet.
    """

    def __init__(self, _):
        logger.info("PyTorch/XLA is enabled. Device is controlled by env var GPU_NUM_DEVICES")
        self.torch_xla = import_module("torch_xla")
        self.amp = import_module("torch_xla.amp")
        self.lm = import_module("torch_xla.core.xla_model")
        device = self.lm.xla_device()
        super().__init__(device)

    def setup_model(self, model):
        model = model.to(device=self.device)
        model.train()
        return (model, model)

    def step(self):
        self.lm.mark_step()

    @property
    def autocast(self):
        return self.amp.autocast

    @staticmethod
    def forward_amp_only():
        return True

    def finalize(self):
        """Print LTC metric report."""
        met = import_module("torch_xla.debug.metrics")
        print(met.metrics_report())

    def profile_latency_start(self):
        """Start profiling latency."""
        raise NotImplementedError

    def profile_latency_stop(self):
        """Stop profiling latency."""
        raise NotImplementedError

    def get_latency_stats(self):
        """Get and reset the latency stats."""
        raise NotImplementedError


class RazorSetter(DeviceModelSetter):
    """The settings for LazyTensorCore with Razor."""

    def __init__(self, options):
        env = self.dump_env(options)

        self.razor = import_module("razor")
        self.lm = import_module("razor.lazy_tensor_core.core.lazy_model")
        self.raf = import_module("raf")
        self.razor_device = self.raf.Device("cpu" if env["RAZOR_DEVICE"] == "CPU" else "cuda")
        super().__init__("lazy")

    @staticmethod
    def dump_env(options):
        """Dump Razor environment information."""
        env_n_default = {
            "RAZOR_DEVICE": "CPU",
            "ENABLE_PARAM_ALIASING": "false",
            "RAZOR_MEMORY_BUDGET": -1,
        }

        if (
            options.get("remat", False)
            and os.environ.get("RAZOR_DEVICE", env_n_default["RAZOR_DEVICE"]) == "GPU"
        ):
            memory_budget = 0.82 * get_gpu_total_memory()
            os.environ["RAZOR_MEMORY_BUDGET"] = str(int(memory_budget * 1e6))

        env = {}
        logger.info("Razor Environment:")
        for env_name, default_value in env_n_default.items():
            val = os.environ.get(env_name, default_value)
            env[env_name] = val
            logger.info("\t%s: %s", env_name, val)

        return env

    def setup_model(self, model):
        model = model.to(device=self.device)
        model.train()
        return (model, self.razor.jit.script(model))

    def step(self):
        self.lm.mark_step()

    @property
    def autocast(self):
        return self.razor.amp.autocast

    def optim(self, name, params, lr=0.1, **kwargs):
        """Wrap the optimizers to use Razor's."""
        if name == "SGD":
            momentum = kwargs.get("momentum", 0.01)
            return self.razor.optimizer.SGD(params, lr=lr, momentum=momentum)
        if name == "Adam":
            eps = kwargs.get("eps", 1e-8)
            return self.razor.optimizer.Adam(params, lr=lr, eps=eps)
        if name == "LANS":
            return self.razor.optimizer.LANS(params)
        raise ValueError("Unsupportde optimizer: " % name)

    @staticmethod
    def forward_amp_only():
        return False

    def finalize(self):
        """Print LTC metric report."""
        met = import_module("razor.lazy_tensor_core.debug.metrics")
        print(met.metrics_report(), flush=True)

    def profile_latency_start(self):
        """Start profiling latency."""
        self.raf.utils.profiler.get()  # Clear existing results if any.
        self.raf.utils.profiler.start(prof_level=2)

    def profile_latency_stop(self):
        """Stop profiling latency."""
        self.raf.utils.profiler.stop()

    def get_latency_stats(self):
        """Get and reset the latency stats."""
        return self.raf.utils.profiler.get()

    def profile_memory_start(self):
        """Start profiling memory."""
        self.raf.utils.memory_profiler.reset()
        self.raf.utils.memory_profiler.start()

    def profile_memory_stop(self):
        """Stop profiling memory."""
        self.raf.utils.memory_profiler.stop()

    def reset_memory_stats(self):
        """Reset memory stats."""
        self.raf.utils.memory_profiler.reset()

    def max_memory_allocated(self, bencher):
        """Get max memory allocated in MBs."""
        mem_info = self.raf.utils.memory_profiler.get_max_memory_info(self.razor_device)
        assert "max_used" in mem_info, "Internal error: max_used is not found in memory info"
        param_size = bencher.param_size
        state_size = 0
        if bencher.optimizer is not None:
            state_size = get_tensor_size(
                [s for v in bencher.optimizer.state.values() for s in v.values()]
            )
        return mem_info["max_used"].value + param_size + state_size

    def max_memory_reserved(self, bencher):
        """Get max memory reserved in MBs."""
        mem_info = self.raf.utils.memory_profiler.get_max_memory_info(self.razor_device)
        assert (
            "max_allocated" in mem_info
        ), "Internal error: max_allocated is not found in memory info"
        param_size = bencher.param_size
        state_size = 0
        if bencher.optimizer is not None:
            state_size = get_tensor_size(
                [s for v in bencher.optimizer.state.values() for s in v.values()]
            )
        return mem_info["max_allocated"].value + param_size + state_size


class TorchBencher(ModelBencherBase):
    """The class to benchmark a PyTorch model.

    model: Any
        The model object from any framework.

    input_shape: Tuple[int, ...]
        The input shape of the model.

    args: List[Any]
        A list of input argument tensors.

    dy: Any
        A tensor of dy (only used in training).

    y_true: Any
        A tensor of y true (only used in training).
    """

    def __init__(self, model, input_shape, args, dy, y_true, **kwargs):
        super().__init__(model, input_shape, args, dy, y_true, **kwargs)
        self.param_size = get_tensor_size(model.parameters())
        self.executor = None
        self.md_setter = None
        self.optimizer = None

    def bench_infer_setup(self, device="cuda", **kwargs):
        self.args = [arg.to(device=device) for arg in self.args]
        self.model.to(device=device)

        # Since we convert PyTorch model to RAF for training,
        # the mathematic expression of PyTorch training model is the one
        # mapped to RAF model.
        self.model.train()

        def _run():
            with torch.no_grad():
                out = self.model(*self.args)
            if device == "cuda":
                torch.cuda.synchronize()
            return out

        self.executor = _run

    def bench_train_setup(self, device="cuda", **kwargs):
        #self.md_setter = DeviceModelSetter.create(kwargs, device)
        #device = self.md_setter.device

        #amp = kwargs.get("amp", False)
        #if amp:
        #    logger.info("AMP enabled in PyTorch.")

        pt_model, self.model = self.md_setter.setup_model(self.model)
        optimizer_name = kwargs.get("optimizer", "SGD")
        self.optimizer = self.md_setter.optim(optimizer_name, pt_model.parameters())

        self.args = [arg.to(device=device) for arg in self.args]
        self.dy = self.dy.to(device=device)
        self.y_true = self.y_true.to(device=device)

        # scalar = torch.cuda.amp.grad_scaler.GradScaler()

        def _run():
            self.optimizer.zero_grad()

            def compute_loss():
                # Wrap with a function to make sure the loss is the only
                # alive tensor after the forward pass.
                with self.md_setter.autocast(amp):
                    t_y = self.model(*self.args)
                    if isinstance(t_y, tuple):
                        t_y = t_y[0]
                    elif isinstance(t_y, dict):
                        assert (
                            "logits" in t_y
                        ), "Expect ModelingOutputs with logits, but got %s" % type(t_y)
                        t_y = t_y["logits"]

                    # Reshape output if necessary.
                    reshape = self.kwargs.get("reshape_output", None)
                    if reshape is not None:
                        t_y = t_y.view(*reshape)
                        y_true = torch.flatten(self.y_true)
                    else:
                        y_true = self.y_true

                    t_ypred = torch.log_softmax(t_y, dim=-1)
                    t_loss = torch.nn.functional.nll_loss(t_ypred, y_true)
                    if isinstance(t_loss, tuple):
                        if hasattr(t_loss[0], "backward"):
                            t_loss = t_loss[0]
                        else:
                            assert hasattr(t_loss[1], "backward")
                            t_loss = t_loss[1]
                return t_loss

            t_loss = compute_loss()

            with self.md_setter.autocast(amp and not self.md_setter.forward_amp_only()):
                # Loss scaling with AMP.
                # FIXME(comaniac): RAF does not have loos scaling yet, so turn off now
                # to make a fair comparison.
                if amp and False:
                    pass
                    # scalar.scale(t_loss).backward()
                    # scalar.step(optimizer)
                    # scalar.update()
                else:
                    t_loss.backward()
                    self.optimizer.step()

                    if optimizer_name == "LANS":
                        self.optimizer.zero_grad()
                    else:
                        # Explicitly clear gradients to let them be optimized in LazyTensor IR.
                        self.optimizer.zero_grad(set_to_none=True)

                self.md_setter.step()
            return t_loss

        self.executor = _run

    def post_warmup(self):
        """Print the current metrics."""
        self.md_setter.finalize()

    def post_bench(self):
        """Print the current metrics."""
        self.md_setter.finalize()

    def profile_latency(self, device="cuda", train=True, **kwargs):
        """Profile the latency of each execution kernel.

        Parameters
        ----------
        device: str
            The target device. Default "cuda".

        train: bool
            When set to True, evaluating the latency of SGD training. Default True.

        kwargs:
            See kwargs in bench_train_setup/bench_infer_setup for details.

        Returns
        -------
        Dict[str, Any]:
            The latency profiling result in JSON format.
        """
        if train:
            self.bench_train_setup(device=device, **kwargs)
        else:
            self.bench_infer_setup(device=device, **kwargs)

        # Warmup.
        timeit.repeat(self.executor, repeat=1, number=10)

        self.md_setter.profile_latency_start()
        self.executor()
        self.md_setter.profile_latency_stop()

        return self.md_setter.get_latency_stats()
