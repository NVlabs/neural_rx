"""Batch-one TensorRT export, inference and coded PUSCH validation.

TensorRT is optional and imported only when building or loading an engine.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from neural_rx.deployment import AerialReceiver


INPUT_NAMES = ("received_real", "received_imag", "ls_channel_real", "ls_channel_imag",
               "active_users", "dmrs_ofdm_indices", "dmrs_subcarrier_indices")


def load_experiment(config_path, checkpoint, *, num_iterations=None,
                    num_prbs=132, device="cuda:0", training_aligned=False):
    """Load full trained weights before selecting an iteration prefix."""
    from neural_rx import NeuralReceiver, SionnaPuschAdapter, load_experiment_config
    from neural_rx.training import load_checkpoint

    device = "cuda:0" if str(device) == "cuda" else str(device)
    experiment = load_experiment_config(config_path)
    adapter = SionnaPuschAdapter.from_config(
        config_path, mode="evaluation", n_size_bwp_override=num_prbs,
        num_users_override=2, device=device,
    )
    receiver = NeuralReceiver(experiment.receiver).to(device).eval()
    metadata = load_checkpoint(receiver, checkpoint, device=device,
                               extra_tensors=adapter.checkpoint_tensors)
    if num_iterations is not None:
        if not 1 <= num_iterations <= len(receiver.iterations):
            raise ValueError("Requested iteration prefix exceeds trained receiver")
        receiver.iterations = receiver.iterations[:num_iterations]
    wrapper = (TrainingAlignedAerialReceiver(receiver, adapter) if training_aligned
               else AerialReceiverWithChannel(receiver, 2))
    return adapter, wrapper.eval(), metadata


class AerialReceiverWithChannel(AerialReceiver):
    """Original two-output deployment boundary, using current trained weights."""

    def forward(self, received_real, received_imag, ls_channel_real, ls_channel_imag,
                active_users, dmrs_ofdm_indices, dmrs_subcarrier_indices):
        received, pilots, channel = self.prepare_inputs(
            received_real, received_imag, ls_channel_real, ls_channel_imag,
            dmrs_ofdm_indices, dmrs_subcarrier_indices,
        )
        result = self.receiver(received, pilots, active_users, channel)[-1]
        return -result.llrs.permute(0, 4, 1, 2, 3).contiguous(), result.channel


class TrainingAlignedAerialReceiver(AerialReceiverWithChannel):
    """Preserve Aerial I/O while matching the trained PyTorch preprocessing.

    The DMRS layout and resource grid are fixed at export. Interpolation uses
    Sionna's global nearest-neighbor map and positional features use the
    reserved-pilot mask used in training.

    Because the grid is fixed, the complete input rearrangement (FOCC pair
    expansion, nearest-neighbor interpolation, subcarrier folding and per-user
    stacking) is a constant permutation. ``forward`` applies it as a single
    gather, which replaces a chain of small layout kernels in the engine;
    ``prepare_inputs`` keeps the step-by-step reference used to derive it.
    """

    def __init__(self, receiver, adapter):
        super().__init__(receiver, adapter.num_users)
        pilots = adapter.transmitter.pilot_pattern.pilots[:, 0]
        nearest = adapter.channel_estimator._interpol._gather_ind[:, 0]
        indices = torch.stack([
            torch.searchsorted(torch.nonzero(p.abs() > 0).flatten(), lookup)
            for p, lookup in zip(pilots, nearest)
        ]).permute(0, 2, 1).contiguous()
        self.register_buffer("nearest_pilot", indices)
        self.register_buffer("trained_pilot_features", adapter.pilot_features.clone())
        self._register_gather(adapter, int((pilots[0].abs() > 0).sum()))

    @staticmethod
    def _average_focc(sparse):
        """Average the two FOCC estimates of every CDM pair: [B,P,U,C] -> [B,P/2,U,C]."""
        batch, pilots, users, components = sparse.shape
        return sparse.reshape(batch, pilots // 2, 2, users, components).mean(dim=2)

    def _interpolate(self, averaged, subcarriers, symbols):
        """Nearest-neighbor interpolate averaged estimates to [B,U,F,T,C]."""
        batch, _, users, components = averaged.shape
        sparse = averaged.repeat_interleave(2, dim=1).permute(0, 2, 1, 3)
        indices = self.nearest_pilot.flatten(1)[None, :, :, None].expand(
            batch, users, -1, components)
        return torch.gather(sparse, 2, indices).reshape(
            batch, users, subcarriers, symbols, components)

    def prepare_inputs(self, received_real, received_imag, ls_channel_real,
                       ls_channel_imag, dmrs_ofdm_indices, dmrs_subcarrier_indices):
        del dmrs_ofdm_indices, dmrs_subcarrier_indices
        received = torch.cat((received_real, received_imag), dim=-1)
        sparse = torch.cat((ls_channel_real, ls_channel_imag), dim=-1)
        channel = self._interpolate(
            self._average_focc(sparse), received.shape[1], received.shape[2])
        return received, self.trained_pilot_features, channel

    def _register_gather(self, adapter, num_pilots):
        """Trace source positions through the reference rearrangement."""
        grid = adapter.resource_grid
        subcarriers, symbols = grid.fft_size, grid.num_ofdm_symbols
        components = 2 * adapter.num_rx_antennas
        users = self.max_num_users
        device = self.nearest_pilot.device
        shapes = ((1, subcarriers, symbols, components),
                  (1, num_pilots // 2, users, components),
                  tuple(self.trained_pilot_features.shape))
        sources, offset = [], 0
        for shape in shapes:  # float64 represents every position exactly
            count = math.prod(shape)
            sources.append(torch.arange(offset, offset + count, device=device,
                                        dtype=torch.float64).reshape(shape))
            offset += count
        received, averaged, pilot_features = sources
        channel = self._interpolate(averaged, subcarriers, symbols)
        features, _ = self.receiver.initializer_features(received, pilot_features, channel)
        _, pilots = self.receiver.initializer_features(
            torch.zeros(shapes[0], device=device), self.trained_pilot_features,
            torch.zeros((1, users, subcarriers, symbols, components), device=device))
        self.feature_shape = tuple(features.shape[1:])
        self.register_buffer("feature_index", features.flatten().round().to(torch.int64))
        self.register_buffer("feature_pilots", pilots.contiguous())
        self.register_buffer("pilot_source", self.trained_pilot_features.flatten()[None].clone())

    def forward(self, received_real, received_imag, ls_channel_real, ls_channel_imag,
                active_users, dmrs_ofdm_indices, dmrs_subcarrier_indices):
        del dmrs_ofdm_indices, dmrs_subcarrier_indices
        received = torch.cat((received_real, received_imag), dim=-1)
        averaged = self._average_focc(torch.cat((ls_channel_real, ls_channel_imag), dim=-1))
        batch, users = active_users.shape
        scale = self.receiver.input_scale(received).reshape(batch, 1)
        source = torch.cat((received.flatten(1) * scale, averaged.flatten(1) * scale,
                            self.pilot_source.expand(batch, -1)), dim=1)
        features = source[:, self.feature_index].reshape(batch * users, *self.feature_shape)
        pilots = self.feature_pilots
        if batch != 1:
            pilots = pilots.repeat(batch, 1, 1, 1)
        result = self.receiver.forward_features(features, pilots, active_users)
        # Aerial expects log(P(bit=0)/P(bit=1)), the negative of the receiver LLRs.
        llrs = -result.llrs.permute(0, 4, 1, 2, 3).contiguous()
        return llrs, result.channel


class AerialDataGenerator:
    """Generate true received slots and nonzero-pilot LS values before FOCC."""

    def __init__(self, adapter):
        self.adapter = adapter
        mask = adapter.transmitter.pilot_pattern.mask[:, 0].bool()
        pilots = adapter.transmitter.pilot_pattern.pilots[:, 0]
        self.nonzero = [torch.nonzero(p.abs() > 0).flatten() for p in pilots]
        self.pilot_values = [p[indices] for p, indices in zip(pilots, self.nonzero)]
        self.coordinates = []
        time_indices, frequency_indices = [], []
        for user, selected in enumerate(self.nonzero):
            coordinates = torch.nonzero(mask[user])[selected]
            self.coordinates.append(coordinates)
            time_indices.append(coordinates[:, 0].unique())
            frequency_indices.append((coordinates[:, 1] % 12).unique())
        self.dmrs_ofdm = torch.stack(time_indices).to(torch.int32)
        self.dmrs_subcarriers = torch.stack(frequency_indices).to(torch.int32)

    def inputs_from_batch(self, batch):
        antennas = self.adapter.num_rx_antennas
        real, imag = batch.received[..., :antennas], batch.received[..., antennas:]
        received = torch.complex(real, imag)
        # Pilot values may change with slot/RNTI; read them after sampling.
        pilot_values = [p[indices] for p, indices in zip(
            self.adapter.transmitter.pilot_pattern.pilots[:, 0], self.nonzero)]
        sparse = torch.stack([
            received[:, coordinates[:, 1], coordinates[:, 0], :] / pilots[None, :, None]
            for coordinates, pilots in zip(self.coordinates, pilot_values)
        ], dim=2)
        return (real.contiguous(), imag.contiguous(), sparse.real.contiguous(),
                sparse.imag.contiguous(), batch.active_users.contiguous(),
                self.dmrs_ofdm, self.dmrs_subcarriers)

    def __call__(self, batch_size=10, ebno_db=6., num_tx=2):
        batch = self.adapter.sample(batch_size, ebno_db, num_active_users=num_tx)
        return self.inputs_from_batch(batch), batch


def export_receiver(wrapper, inputs, destination):
    """Export a static batch-one, configured-grid two-output ONNX graph."""
    if inputs[0].shape[0] != 1:
        raise ValueError("The real-time engine requires batch size one")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(wrapper.eval(), inputs, destination, dynamo=True,
                          opset_version=18, input_names=list(INPUT_NAMES),
                          output_names=["llrs", "h_hat"], external_data=False)


def _autocast_fp16(onnx_path, calibration_inputs=None):
    """Cast to FP16 with ModelOpt, keeping FP32 I/O for the runner."""
    import tempfile

    from modelopt.onnx.autocast import convert_to_mixed_precision

    with tempfile.TemporaryDirectory() as directory:
        calibration_data = None
        if calibration_inputs is not None:
            calibration_data = str(Path(directory) / "calibration.npz")
            np.savez(calibration_data, **{
                name: value.detach().cpu().numpy()
                for name, value in zip(INPUT_NAMES, calibration_inputs)})
        model = convert_to_mixed_precision(
            str(onnx_path), low_precision_type="fp16", keep_io_types=True,
            calibration_data=calibration_data)
    return model.SerializeToString()


def build_engine(onnx_path, engine_path, *, fp16=True, workspace_bytes=2 << 30,
                 calibration_inputs=None, optimization_level=None, timing_iterations=None):
    """Build on this GPU; timing and serialized engines are platform-specific.

    TensorRT 11 builds strongly typed networks only, so FP16 is applied to the
    ONNX graph with ModelOpt. ``calibration_inputs`` provides valid sample inputs
    for ModelOpt's reference run when random inputs are not valid.
    ``optimization_level`` (0-5, TensorRT default 3) trades build time for a
    more exhaustive tactic search; ``timing_iterations`` averages the tactic
    timing over several runs, which makes the selected kernels reproducible.
    """
    import tensorrt as trt

    onnx_bytes = Path(onnx_path).read_bytes()
    fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
    if fp16 and fp16_flag is None:
        onnx_bytes = _autocast_fp16(onnx_path, calibration_inputs)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_bytes):
        raise RuntimeError("\n".join(str(parser.get_error(i))
                                     for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    if fp16 and fp16_flag is not None:
        config.set_flag(fp16_flag)
    if optimization_level is not None:
        config.builder_optimization_level = optimization_level
    if timing_iterations is not None:
        config.avg_timing_iterations = timing_iterations
    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the receiver engine")
    Path(engine_path).write_bytes(bytes(serialized))
    return {"build_seconds": time.perf_counter() - start,
            "tensorrt": trt.__version__, "fp16_enabled": fp16,
            "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda}


class TensorRTRunner:
    """Run TensorRT with CUDA tensors, keeping transfers outside timed inference.

    Engine metadata (names, shapes, dtypes) and the output bindings are
    resolved once at construction, so a call only validates and binds the
    input tensors before enqueueing the engine.
    """

    def __init__(self, engine_path):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError("Could not deserialize TensorRT engine")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.input_names, self.output_names = [], []
        self.outputs = {}
        self._input_specs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(
                self.engine.get_tensor_dtype(name)))).dtype
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                self._input_specs.append((INPUT_NAMES.index(name), name, shape, dtype))
            else:
                self.output_names.append(name)
                self.outputs[name] = torch.empty(shape, dtype=dtype, device="cuda")
                self.context.set_tensor_address(name, self.outputs[name].data_ptr())

    def __call__(self, inputs):
        self._inputs = []  # keep bound tensors alive until the next call
        for index, name, shape, dtype in self._input_specs:
            value = inputs[index]
            if not value.is_contiguous():
                value = value.contiguous()
            if tuple(value.shape) != shape:
                raise ValueError(f"{name}: expected {shape}, got {tuple(value.shape)}")
            if not value.is_cuda or value.dtype != dtype:
                raise ValueError(f"{name}: expected CUDA {dtype}")
            self._inputs.append(value)
            self.context.set_tensor_address(name, value.data_ptr())
        current = torch.cuda.current_stream()
        self.stream.wait_stream(current)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        current.wait_stream(self.stream)
        return self.outputs


def benchmark(runner, inputs, *, warmup=20, repetitions=100):
    """Report steady-state batch-one CUDA-stream and synchronized wall latency."""
    for _ in range(warmup):
        runner(inputs)
    torch.cuda.synchronize()
    elapsed, wall = [], []
    for _ in range(repetitions):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        runner(inputs)
        stop.record()
        stop.synchronize()
        elapsed.append(start.elapsed_time(stop))
        wall.append((time.perf_counter() - begin) * 1000)
    return {"gpu": torch.cuda.get_device_name(), "batch_size": 1,
            "repetitions": repetitions, "cuda_stream_ms_median": float(np.median(elapsed)),
            "cuda_stream_ms_p95": float(np.percentile(elapsed, 95)),
            "synchronized_wall_ms_median": float(np.median(wall))}


@torch.no_grad()
def validate_coded_bler(runner, generator, *, num_iter=10, batch_size=10, ebno_db=6.,
                        eager_wrapper=None):
    """Original 100-slot experiment: generate batches, infer one slot, FEC-decode."""
    bit_errors = block_errors = num_bits = num_blocks = 0
    eager_bit_errors = eager_block_errors = native_bit_errors = native_block_errors = 0
    max_llr_error = 0.
    for i in range(num_iter):
        inputs, batch = generator(batch_size, ebno_db, 2)
        llrs = []
        for j in range(batch_size):
            slot = tuple(t[j:j+1] if k < 5 else t for k, t in enumerate(inputs))
            llrs.append(runner(slot)["llrs"].clone())
        llrs = torch.cat(llrs)
        decoded, _ = generator.adapter.decode(-llrs.permute(0, 2, 3, 4, 1))
        target = batch.information_bits
        errors = decoded != target
        bit_errors += int(errors.sum())
        block_errors += int(errors.any(dim=-1).sum())
        num_bits += target.numel()
        num_blocks += target.shape[0] * target.shape[1]
        if eager_wrapper is not None:
            eager, _ = eager_wrapper(*inputs)
            max_llr_error = max(max_llr_error, float((eager - llrs).abs().max()))
            eager_decoded, _ = generator.adapter.decode(-eager.permute(0, 2, 3, 4, 1))
            eager_errors = eager_decoded != target
            eager_bit_errors += int(eager_errors.sum())
            eager_block_errors += int(eager_errors.any(dim=-1).sum())
            native_llrs = eager_wrapper.receiver.forward_llrs(
                batch.received, batch.pilot_features, batch.active_users,
                batch.initial_channel,
            )
            native_decoded, _ = generator.adapter.decode(native_llrs)
            native_errors = native_decoded != target
            native_bit_errors += int(native_errors.sum())
            native_block_errors += int(native_errors.any(dim=-1).sum())
        print(f"Progress {(i+1)/num_iter:.0%}; BER={bit_errors/num_bits:.6f}, "
              f"TBLER={block_errors/num_blocks:.4f}", flush=True)
    result = dict(ebno_db=ebno_db, slots=num_iter*batch_size, num_bits=num_bits,
                  num_blocks=num_blocks, bit_errors=bit_errors, block_errors=block_errors,
                  ber=bit_errors/num_bits, bler=block_errors/num_blocks)
    if eager_wrapper is not None:
        result.update(eager_ber=eager_bit_errors/num_bits,
                      eager_bler=eager_block_errors/num_blocks,
                      native_ber=native_bit_errors/num_bits,
                      native_bler=native_block_errors/num_blocks,
                      max_abs_llr_error=max_llr_error)
    return result
