"""
The module of wrapper classes to benchmark RAF models.
"""
# pylint: disable=too-many-arguments, unused-argument, protected-access,no-member,not-callable, inconsistent-return-statements
from pathlib import Path
import timeit
import re

import tvm
import raf
from raf._core.executor import VMExecutor
from raf.model.trace import _get_func_inputs
from raf.testing import check, numpy
from raf.utils.tuner import run_tuning
from raf import distributed as dist

from logger import get_logger
from model_bencher import ModelBencherBase
from utils import get_gpu_total_memory, get_tensor_size

logger = get_logger("RAF")  # pylint: disable=invalid-name


class RAFBencher(ModelBencherBase):
    """The class to benchmark a RAF model.

    model: raf.ModelBase
        The model object of RAF.

    input_shape: Tuple[int, ...]
        The input shape of the model.

    args: List[raf.ndarray]
        A list of input arguments.

    dy: Any
        A tensor of dy (only used in training).

    y_true: Any
        A tensor of y true (only used in training).

    ref_bencher: Optional[ModelBencherBase]
        Model bencher of the reference model.
    """

    def __init__(self, raf_model, input_shape, args, dy, y_true, ref_bencher=None, **kwargs):
        super().__init__(raf_model, input_shape, args, dy, y_true, **kwargs)
        self.ref_bencher = ref_bencher
        reshape = self.kwargs.get("reshape_output", None)

        self.model_w_loss = None
        if self.model is not None:
            if hasattr(self.model, "record"):
                out = self.model.record(*self.args)

                # Reshape output if necessary.
                if reshape is not None:
                    out = raf._op.sym.reshape(out, reshape)
                    y_true = raf._op.sym.reshape(self.y_true, [-1])

                y_true = raf._op.sym.reshape(self.y_true, [-1])
                y_pred = raf._op.sym.log_softmax(out)
                loss = raf._op.sym.nll_loss(y_true, y_pred)

                self.model_w_loss = self.model + loss
            else:
                logger.info("Skip appending loss function because RAF model is not FrameworkModel")
                self.model_w_loss = self.model

        self.tvm_device = None
        self.vm_inputs = None
        self.vm = None
        self.trainer = None
        self.data_parallel = None
        self.zero_opt = None
        self.device = None

    @staticmethod
    def validate_sch_file(sch_file):
        """Validate the given schedule file and print a warning if the schedule file path
        is given but the file does not exist.

        Parameters
        ----------
        sch_file: Union[str, None]
            The path of schedule file or None.
        """
        if sch_file is not None:
            sch_file_path = Path(sch_file)
            if not sch_file_path.exists():
                logger.warning("Schedule file not found: %s", sch_file_path.absolute())

    def get_vm_exec(
        self,
        model,
        args,
        device,
        disable_fuse=False,
        sch_file=None,
        dryrun=False,
        amp=False,
        remat=False,
    ):
        """Helper function to initialize a VM to save to self.vm_inpus and self.vm_exec.

        Parameters
        ----------
        model: raf.ModelBase
            The model object of RAF.

        args: List[raf.ndarray]
            A list of input arguments.

        device: str
            The target device. Default "cuda".

        disable_fuse: bool
            Whether to disable fusion.

        sch_file: str
            The log file of tuning records.

        dryrun: bool
            Whether to dryrun (for tuning).

        amp: bool
            Whether to use AMP.

        remat: bool
            Whether to enable rematerialization. If so, the memory budget is set to 90%
            of the total memory.

        Returns
        -------
        exec: Callable
            A function to execute the model using VM.
        """
        record = model._internal(*args)
        mod = record.mod
        config = {}
        disabled_pass = ["AutoSchedulerLayoutRewrite"]
        if disable_fuse:
            disabled_pass += ["FuseDialect", "FuseTVM"]
        if amp:
            logger.info("AMP enabled in RAF.")
            mod = raf._ffi.pass_.AutoCast()(mod)
        self.vm_inputs = _get_func_inputs(record, args, {}, get_handle=False)

        if remat:
            if device.startswith("cuda"):
                config["raf.memory_budget"] = int(0.9 * get_gpu_total_memory() * 1e6)
                config["raf.remat.use_gflops_cost"] = False
                logger.info("Rematerialization enabled in RAF")
            else:
                logger.warning("Rematerialization only enabled on GPU for now")

        with tvm.transform.PassContext(
            opt_level=3,
            config=config,
            disabled_pass=disabled_pass,
        ):
            self.vm = VMExecutor(mod, device, dryrun=dryrun)
            vm_exec = self.vm.make_executor(sch_file)
            return lambda: vm_exec(*self.vm_inputs)

    def bench_infer_setup(self, device="cuda", **kwargs):
        """Setup the model to benchmark forward inference.

        Parameters
        ----------
        device: str
            The target device. Default "cuda".

        kwargs:
            use_interpreter: bool
                Use interpreter instead of VM to benchmerk.
            disable_fuse: bool
                Whether to disable fusion.
            sch_file: str
                The tuned log file path.
            dryrun: bool
                Whether to dryrun (for tuning).
        """
        self.model.infer_mode()
        self.data_parallel = kwargs.get("data_parallel", False)
        self.zero_opt = kwargs.get("zero_opt", 0)
        self.device = device
        assert (
            not self.data_parallel and self.zero_opt == 0
        ), "Doesn't support data parallel or ZeRO in infer mode"
        self.tvm_device = tvm.nd.device(device)
        self.args = [arg.to(device=device) for arg in self.args]
        self.model.to(device=device)

        dryrun = kwargs.get("dryrun", False)
        disable_fuse = kwargs.get("disable_fuse", False)
        amp = kwargs.get("amp", False)
        remat = kwargs.get("remat", False)

        if "use_interpreter" in kwargs and kwargs["use_interpreter"]:
            assert not dryrun, "Dryrun is only available on VM"

            def run_interpreter():
                out = self.model(*self.args)
                self.tvm_device.sync()
                return out

            self.executor = run_interpreter
        else:
            sch_file = kwargs.get("sch_file", None)
            self.validate_sch_file(sch_file)
            vm_exec = self.get_vm_exec(
                self.model,
                self.args,
                device,
                disable_fuse,
                sch_file,
                dryrun,
                amp,
                remat,
            )

            def _run_vm():
                out = vm_exec()
                self.tvm_device.sync()
                if isinstance(out, (raf._core.value.TupleValue, tuple, list)):
                    return out[0]
                return out

            self.executor = _run_vm

    def bench_train_setup(self, device="cuda", **kwargs):
        """Setup the model to benchmark backward training.

        Parameters
        ----------
        device: str
            The target device. Default "cuda".

        kwargs:
            use_interpreter: bool
                Use interpreter instead of VM to benchmerk.
            disable_fuse: bool
                Whether to disable fusion.
            amp: bool
                Whether use automatic mixed precision (AMP).
            sch_file: str
                The tuned log file path.
            dryrun: bool
                Whether to dryrun (for tuning).
            data_parallel: bool
                Whether to enable data (activation) parallel.
            zero_opt: int
                The ZeRO optimization level.
            optimizer: str
                The optimizer. option: SGD and LANS
        """
        # pylint: disable=too-many-statements
        self.model_w_loss.train_mode()

        # Process kwargs.
        self.data_parallel = kwargs.get("data_parallel", False)
        self.zero_opt = kwargs.get("zero_opt", 0)
        dryrun = kwargs.get("dryrun", False)
        disable_fuse = kwargs.get("disable_fuse", False)
        amp = kwargs.get("amp", False)
        remat = kwargs.get("remat", False)
        optimizer = kwargs.get("optimizer", "SGD")

        if self.data_parallel or self.zero_opt > 0:
            if device != "cuda":
                raise RuntimeError("Only support data parallel or ZeRO on CUDA")
            if not raf.build.with_distributed():
                raise RuntimeError("RAF is not built with MNM_USE_MPI=ON and MNM_USE_NCCL=ON")

            comm = dist.get_communicator()
            dctx = dist.get_config()
            dctx.enable_data_parallel = self.data_parallel
            dctx.zero_opt_level = self.zero_opt
            local_rank = comm.local_rank
            device = f"cuda({local_rank})"
        self.device = device

        if self.device.find("(") != -1:
            device_type = self.device.split("(")[0]
            device_id = int(self.device.split("(")[-1].split(")")[0])
            self.tvm_device = tvm.nd.device(device_type, device_id)
        else:
            self.tvm_device = tvm.nd.device(self.device)

        self.args = [arg.to(device=self.device) for arg in self.args]
        for arg in self.args:
            arg.requires_grad = True
        if isinstance(self.dy, list):
            self.dy = tuple([y.to(device=self.device) for y in self.dy])
        else:
            self.dy = self.dy.to(device=self.device)
        self.y_true = self.y_true.to(device=self.device)
        self.model_w_loss.to(device=self.device)

        if "use_interpreter" in kwargs and kwargs["use_interpreter"]:
            assert not dryrun, "Dryrun is only available on VM"

            def run_interpreter():
                loss = self.model_w_loss(*self.args, self.y_true)
                loss.backward()
                self.tvm_device.sync()
                return loss

            self.executor = run_interpreter
        else:
            if optimizer == "SGD":
                self.trainer = raf.optim.sgd.with_sgd(learning_rate=0.1, momentum=0.01)(
                    self.model_w_loss
                )
            elif optimizer == "LANS":
                self.trainer = raf.optim.lans.with_lans()(self.model_w_loss)
            else:
                assert False, "only support SGD and LANS for now"
            sch_file = kwargs.get("sch_file", None)
            self.validate_sch_file(sch_file)

            vm_exec = self.get_vm_exec(
                self.trainer,
                [self.dy, *self.args, self.y_true],
                self.device,
                disable_fuse,
                sch_file,
                dryrun,
                amp,
                remat,
            )

            def run_vm():
                loss = vm_exec()
                self.tvm_device.sync()
                while isinstance(loss, (tuple, tvm.ir.container.Array, raf._core.value.TupleValue)):
                    loss = loss[0]
                return loss

            self.executor = run_vm

    def post_bench(self):
        """Reset distributed context settings after benchmark."""
        dctx = dist.get_config()
        dctx.enable_data_parallel = False
        dctx.zero_opt_level = 0

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
        f_setup = self.bench_train_setup if train else self.bench_infer_setup(self)
        f_setup(device=device, **kwargs)

        # Warmup.
        timeit.repeat(self.executor, repeat=1, number=10)

        comm = dist.get_communicator()
        if comm.local_rank == 0:
            raf.utils.profiler.get()  # Clear existing results if any.
            raf.utils.profiler.start(prof_level=2)
            self.executor()
            raf.utils.profiler.stop()
            return raf.utils.profiler.get()

        # Only profile on rank 0, so other devices just execute the model as usual.
        self.executor()

    def profile_memory(self, show_used, device="cuda", train=True, **kwargs):
        """Profile the peak memory footprint.

        Parameters
        ----------
        show_used: bool
            Show the total used memory in MBs. If False, then it shows the total allocated
            memory.

        device: str
            The target device. Default "cuda".

        train: bool
            When set to True, evaluating the latency of SGD training. Default True.

        kwargs:
            See kwargs in bench_train_setup/bench_infer_setup for details.

        Returns
        -------
        float:
            The memory footprint in MBs.
        """
        f_setup = self.bench_train_setup if train else self.bench_infer_setup
        kwargs["dryrun"] = True
        f_setup(device=device, **kwargs)

        param_mem = get_tensor_size(self.vm_inputs)

        raf.utils.memory_profiler.reset()
        raf.utils.memory_profiler.start()
        timeit.repeat(self.executor, repeat=2, number=10)
        raf.utils.memory_profiler.stop()

        ret_map = raf.utils.memory_profiler.get_max_memory_info(raf.Device(self.device))
        max_mem_info = {k: v.value for k, v in ret_map.items()}

        print("Memory Profiling Summary")
        print("Parameter: %.2f MBs" % param_mem)
        print("Memory Pool (peak used): %.2f MBs" % max_mem_info["max_used"])
        print("Memory Pool (peak allocated): %.2f MBs" % max_mem_info["max_allocated"])
        print("#GarbageCollection: %.0f" % max_mem_info["num_gc"])

        return param_mem + (
            max_mem_info["max_used"] if show_used else max_mem_info["max_allocated"]
        )

    def get_memory_trace(self):
        """Get the memory trace. Note that trace is available after memory profiling,
        so make sure running profile_memory before calling this function.

        Returns
        -------
        str:
            The memory trace.
        """
        return raf.utils.memory_profiler.get_memory_trace(raf.Device(self.device))

    def check_gradient(self):
        """Check each gradient between RAF and PyTorch model."""
        import torch  # pylint: disable=import-outside-toplevel

        m_model = self.model_w_loss
        t_model = self.ref_bencher.model

        assert m_model is not None
        assert isinstance(t_model, torch.nn.Module), "Only support PyTorch models"

        # Get all parameters from RAF model.
        m_params = [p for p in dir(m_model) if p.startswith("model_")]

        stats = []
        for m_param in m_params:
            # Get the corresponding parameter in PyTorch model.
            grad = t_model
            attr = ""

            for token in m_param[6:].split("_"):
                if attr:
                    attr += "_"
                attr += token

                if hasattr(grad, attr):
                    grad = getattr(grad, attr)
                    attr = ""

            assert isinstance(
                grad, torch.Tensor
            ), "Expected torch.Tensor but got %s. Unmatched attr token: %s" % (
                type(grad),
                attr,
            )

            try:
                check(getattr(m_model, m_param), grad, atol=1e-4, rtol=1e-4)
                stats.append("%s ... passed" % m_param)
            except Exception as err:  # pylint: disable=broad-except
                atol = re.search(r"Max absolute difference: (.+)\n", str(err)).group(1)
                stats.append("%s ... failed (atol %s)" % (m_param, atol))

        for stat in sorted(stats):
            print(stat)

    def check_correctness(self, device="cuda", train=True, check_gradient=False, **kwargs):
        """Check the correctness of the RAF model against to the reference model.

        Notes
        -----
        Correctness checking requires to run both RAF and reference models and both
        models will reserve device memory, so you might encounter out of memory error
        with large batch size. It is recommended to use batch size 1 in correctness checking.

        Parameters
        ----------
        device: str
            The target device. Default "cuda".

        train: bool
            When set to True, check the correctness of SGD training. Default True.

        check_gradient: bool
            Whether to check every gradient value.

        kwargs:
            sch_file: str
                The tuned log file path.

        Returns
        -------
        loss_pair: Tuple[float, float]
            A pair of loss from both RAF and original model.
        """
        assert self.ref_bencher is not None, "Reference bencher is required to check correctness"

        if train:
            self.bench_train_setup(device, **kwargs)
            out = self.bench_stmt()

            self.ref_bencher.bench_train_setup("cpu", **kwargs)
            ref_out = self.ref_bencher.bench_stmt()
        else:
            self.bench_infer_setup(device, **kwargs)
            out = self.bench_stmt()

            self.ref_bencher.bench_infer_setup("cpu", **kwargs)
            ref_out = self.ref_bencher.bench_stmt()

        try:
            check(out, ref_out, rtol=1e-4, atol=1e-4)
            logger.info("Correctness checking success, atol < 1e-4")
        except Exception as err:  # pylint: disable=broad-except
            atol = re.search(r"Max absolute difference: (.+)\n", str(err))
            shape = re.search(r"\(shapes (.+) mismatch\)\n", str(err))
            if atol:
                atol = atol.group(1)
                logger.info("Correctness checking failure, atol %s", str(atol))
            elif shape:
                shape = shape.group(1)
                logger.info("Correctness checking failure, shape mismatch: %s", str(shape))
            else:
                logger.info("Correctness checking failure, unknown error:")
                logger.info(err)

        if check_gradient:
            assert train, "Cannot check gradient for forward inference"
            self.check_gradient()

        return (numpy(out), numpy(ref_out))

    def tune(
        self,
        sch_file,
        device="cuda",
        train=True,
        n_trials=lambda l: 300 * min(l, 100),
        only_tune_tasks_with_name=None,
        only_extract_tasks=False,
        mip_size=1,
        **kwargs,
    ):
        """Use TVM auto-scheduler to tune the model.

        Parameters
        ----------
        sch_file: str
            The log file to dump the tuning records. If the file already contains tuning records,
            we use them to initialize the task scheduler and new records will be appended.

        device: str
            The target device. Default "cuda".

        train: bool
            When set to True, tuning the AutoDiff model. Default True.

        n_trials: Callable[[int], int] or int
            An integer of total number of measurement trials, or a function that determines
            the total number of measurement trials by taking the task number.
            Default is at maximum 30k = 300 * min(100, task_num).

        only_tune_tasks_with_name: Optional[List[str]]
            When specify with a list of name tokens, only the tasks with the tokens in their names
            will be tuned.

        only_extract_tasks: bool
            Whether to extract and print tasks only without actual tuning them.

        mip_size: int
            MIP rank size for data parallel workloads. It's 1 on sigel device
        kwargs:
            Used to setup the model to be tuned. See kwargs in `bench_train_setup` for details.
        """
        if "use_interpreter" in kwargs:
            assert not kwargs["use_interpreter"], "Only support tuning with VM but not interpreter"
        if mip_size != 1:
            comm = dist.get_communicator()
            comm.size = mip_size
        # Use memory profiler to extract tuning tasks because memory profiler
        # will ignore the op compilation failture due to no valid schedule and
        # #skip the op execution.
        f_setup = self.bench_train_setup if train else self.bench_infer_setup
        f_setup(device=device, dryrun=True, **kwargs)

        run_tuning(
            self.vm,
            device,
            self.vm_inputs,
            sch_file,
            n_trials=n_trials,
            only_tune_tasks_with_name=only_tune_tasks_with_name,
            only_extract_tasks=only_extract_tasks,
        )
