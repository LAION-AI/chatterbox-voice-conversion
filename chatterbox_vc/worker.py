"""
Subprocess voice conversion worker.

This module implements a long-running subprocess that loads ChatterboxVC once
and processes conversion requests via a JSON-line protocol over stdin/stdout.
This pattern is useful when:

- The VC model lives in a separate Python environment (e.g., different
  torch version or virtualenv) from the main application.
- You want to isolate GPU memory usage to a specific process.
- You need to run VC on a different GPU than the main application.

Protocol:
    The worker reads one JSON object per line from stdin and writes one
    JSON response per line to a dedicated file descriptor (fd 3).

    Request format::

        {"source": "/path/to/source.wav", "target": "/path/to/target.wav", "output": "/path/to/output.wav"}

    Response format (success)::

        {"status": "ok", "output": "/path/to/output.wav", "sample_rate": 24000, "elapsed": 2.34}

    Response format (error)::

        {"status": "error", "error": "Error message", "traceback": "..."}

Usage:
    Start the worker as a subprocess::

        python -m chatterbox_vc.worker --device cuda:0

    Then send JSON requests via stdin, read responses from fd 3.

    Or use the WorkerClient helper from your main process::

        from chatterbox_vc.worker import WorkerClient
        client = WorkerClient(device="cuda:0")
        result = client.convert("source.wav", "target.wav", "output.wav")
        client.shutdown()
"""

import json
import os
import sys
import subprocess
import time
import traceback
from typing import Optional


def run_worker(device: str = "cuda:0"):
    """Main loop for the subprocess worker.

    Loads ChatterboxVC once, then processes JSON-line requests from stdin.
    Responses are written to fd 3 (must be opened by the parent process).

    Args:
        device: PyTorch device string for model loading.
    """
    # Redirect stdout to stderr so library print() calls don't corrupt
    # the JSON protocol. We use fd 3 for clean protocol output.
    proto_fd = 3
    try:
        proto_out = os.fdopen(proto_fd, "w")
    except OSError:
        # If fd 3 isn't available, fall back to stdout (simpler usage).
        proto_out = sys.stdout
        sys.stdout = sys.stderr

    def send(obj):
        proto_out.write(json.dumps(obj) + "\n")
        proto_out.flush()

    # Load model
    try:
        from chatterbox_vc.convert import VoiceConverter
        vc = VoiceConverter(device=device)
        send({"status": "ready", "device": device})
    except Exception as e:
        send({"status": "error", "error": f"Failed to load model: {e}"})
        sys.exit(1)

    # Process requests
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            send({"status": "error", "error": f"Invalid JSON: {e}"})
            continue

        source = req.get("source")
        target = req.get("target")
        output = req.get("output")

        if not all([source, target, output]):
            send({"status": "error", "error": "Missing required fields: source, target, output"})
            continue

        t0 = time.time()
        try:
            vc.convert(source, target, output)
            elapsed = round(time.time() - t0, 3)
            send({
                "status": "ok",
                "output": output,
                "sample_rate": vc.sample_rate,
                "elapsed": elapsed,
            })
        except Exception as e:
            send({
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })


class WorkerClient:
    """Client for communicating with a VC worker subprocess.

    Spawns a subprocess running the VC worker and provides a simple
    synchronous API for voice conversion requests.

    Args:
        device: PyTorch device for the worker (e.g., "cuda:0").
        python_executable: Path to the Python interpreter to use.
            Useful when the worker needs a different virtualenv.
        timeout: Maximum seconds to wait for a conversion. Default 300.

    Example::

        client = WorkerClient(device="cuda:0")

        # Wait for model to load
        print("Worker ready!")

        # Convert audio
        result = client.convert("source.wav", "target.wav", "output.wav")
        print(f"Done in {result['elapsed']}s")

        # Clean up
        client.shutdown()
    """

    def __init__(
        self,
        device: str = "cuda:0",
        python_executable: str = sys.executable,
        timeout: float = 300,
    ):
        self.timeout = timeout

        # Spawn worker with fd 3 for protocol output
        self._proc = subprocess.Popen(
            [python_executable, "-m", "chatterbox_vc.worker", "--device", device],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Capture library logs
            text=True,
            pass_fds=(3,) if sys.platform != "win32" else (),
        )

        # Read the "ready" message
        ready_line = self._proc.stdout.readline()
        if not ready_line:
            stderr = self._proc.stderr.read()
            raise RuntimeError(f"Worker failed to start: {stderr}")

        ready = json.loads(ready_line)
        if ready.get("status") != "ready":
            raise RuntimeError(f"Worker error: {ready.get('error', 'unknown')}")

    def convert(
        self,
        source: str,
        target: str,
        output: str,
    ) -> dict:
        """Send a conversion request to the worker.

        Args:
            source: Path to source audio WAV file.
            target: Path to target speaker WAV file.
            output: Path to write the output WAV file.

        Returns:
            Response dict with ``status``, ``output``, ``sample_rate``,
            and ``elapsed`` keys.

        Raises:
            RuntimeError: If the worker returns an error or times out.
        """
        request = json.dumps({"source": source, "target": target, "output": output})
        self._proc.stdin.write(request + "\n")
        self._proc.stdin.flush()

        response_line = self._proc.stdout.readline()
        if not response_line:
            raise RuntimeError("Worker process terminated unexpectedly")

        result = json.loads(response_line)
        if result.get("status") == "error":
            raise RuntimeError(f"VC failed: {result.get('error')}")

        return result

    def shutdown(self):
        """Gracefully shut down the worker subprocess."""
        if self._proc and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.wait(timeout=10)

    def __del__(self):
        self.shutdown()


def main():
    """Entry point for ``python -m chatterbox_vc.worker``."""
    import argparse
    parser = argparse.ArgumentParser(description="ChatterboxVC worker subprocess")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device (default: cuda:0)")
    args = parser.parse_args()
    run_worker(device=args.device)


if __name__ == "__main__":
    main()
