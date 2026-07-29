import time
import threading
import multiprocessing as mp
import torch

from model import (
    MyModel,
    LearnInput,
    MODEL_D_MODEL,
    MODEL_NUM_HEADS,
    MODEL_D_FEEDFORWARD,
    MODEL_NUM_LAYERS_ENCODER,
    MODEL_NUM_LAYERS_DECODER,
)

# League model cache — avoid re-loading the same checkpoint from disk every game
_league_cache = {}  # {path: model}


def get_league_model(path, device):
    """Load a league opponent model, caching recently used checkpoints."""
    if path not in _league_cache:
        m = MyModel(
            MODEL_D_MODEL,
            MODEL_NUM_HEADS,
            MODEL_D_FEEDFORWARD,
            MODEL_NUM_LAYERS_ENCODER,
            MODEL_NUM_LAYERS_DECODER
        ).to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            m.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            m.load_state_dict(checkpoint, strict=False)
        m.eval()
        # Keep cache small — only last 3 opponents
        if len(_league_cache) >= 3:
            oldest = next(iter(_league_cache))
            del _league_cache[oldest]
        _league_cache[path] = m
    return _league_cache[path]


class GPUInferenceServer:
    """Collects and batches NN evaluations from parallel game workers."""
    def __init__(self, model, parent_conns, model_lock, batch_size=64, timeout=0.005):
        self.model = model
        self.parent_conns = parent_conns
        self.model_lock = model_lock
        self.batch_size = batch_size
        self.timeout = timeout
        self.running = True
        self.thread = None

    def start(self, device):
        self.thread = threading.Thread(target=self._loop, args=(device,))
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _loop(self, device):
        while self.running:
            try:
                ready_conns = mp.connection.wait(self.parent_conns, timeout=self.timeout)
            except OSError:
                # Catch WinError 1450 (Insufficient system resources on Windows named pipe I/O)
                time.sleep(0.01)
                continue
            if not ready_conns or not self.running:
                continue

            batch_conns = []
            batch_sv_enc = []
            batch_sv_dec = []

            max_infer_batch = 32 if device.type == 'cuda' else self.batch_size
            for conn in ready_conns:
                if len(batch_conns) >= max_infer_batch:
                    break
                try:
                    if conn.poll():
                        sv_enc, sv_dec = conn.recv()
                        batch_conns.append(conn)
                        batch_sv_enc.append(sv_enc)
                        batch_sv_dec.append(sv_dec)
                except (EOFError, OSError):
                    if conn in self.parent_conns:
                        self.parent_conns.remove(conn)

            if not batch_conns:
                continue

            input_enc = LearnInput()
            for sv in batch_sv_enc:
                input_enc.add(sv)

            orig_lens = []
            input_dec = LearnInput()
            for sv in batch_sv_dec:
                orig_len = len(sv.offset)
                orig_lens.append(orig_len)
                if orig_len < 256:
                    for _ in range(256 - orig_len):
                        sv.offset.append(len(sv.index))
                input_dec.add(sv)

            try:
                with self.model_lock:
                    with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')), torch.inference_mode():
                        out_enc, out_dec = self.model(
                            torch.tensor(input_enc.index, dtype=torch.int32, device=device),
                            torch.tensor(input_enc.value, dtype=torch.float32, device=device),
                            torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
                            torch.tensor(input_dec.index, dtype=torch.int32, device=device),
                            torch.tensor(input_dec.value, dtype=torch.float32, device=device),
                            torch.tensor(input_dec.offset, dtype=torch.int32, device=device)
                        )
                        values = out_enc.squeeze(-1).tolist()
                        policies = out_dec.tolist()
                        del out_enc, out_dec
            except Exception as e:
                if device.type == 'cuda':
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                # CPU fallback on CUDA Out Of Memory
                try:
                    cpu_dev = torch.device("cpu")
                    with self.model_lock:
                        with torch.inference_mode():
                            self.model.to(cpu_dev)
                            out_enc, out_dec = self.model(
                                torch.tensor(input_enc.index, dtype=torch.int32, device=cpu_dev),
                                torch.tensor(input_enc.value, dtype=torch.float32, device=cpu_dev),
                                torch.tensor(input_enc.offset, dtype=torch.int32, device=cpu_dev),
                                torch.tensor(input_dec.index, dtype=torch.int32, device=cpu_dev),
                                torch.tensor(input_dec.value, dtype=torch.float32, device=cpu_dev),
                                torch.tensor(input_dec.offset, dtype=torch.int32, device=cpu_dev)
                            )
                            values = out_enc.squeeze(-1).tolist()
                            policies = out_dec.tolist()
                            del out_enc, out_dec
                            self.model.to(device)
                except Exception:
                    try:
                        self.model.to(device)
                    except Exception:
                        pass
                    values = [0.0] * len(batch_conns)
                    policies = [[0.0] * 256] * len(batch_conns)

            for idx, conn in enumerate(batch_conns):
                val = values[idx]
                pol = policies[idx][:orig_lens[idx]]
                try:
                    conn.send((val, pol))
                except (OSError, IOError):
                    pass
