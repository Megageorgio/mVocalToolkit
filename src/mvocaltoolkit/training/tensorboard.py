"""An optional local TensorBoard viewer, with one process owned by the toolkit."""
import asyncio
import os
import socket
import time
from pathlib import Path

import httpx


class TensorBoardViewer:
    def __init__(self):
        self.process = None
        self.run_id = ""
        self.url = ""
        self.lock = asyncio.Lock()

    async def _stop(self):
        process, self.process = self.process, None
        self.run_id, self.url = "", ""
        if process is not None and process.returncode is None:
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), 5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass

    async def stop(self):
        async with self.lock:
            await self._stop()

    async def start(self, run_id: str, folder: Path, python: Path, cwd: Path, env: dict):
        async with self.lock:
            if self.process is not None and self.process.returncode is None and self.run_id == run_id:
                return {"url": self.url, "pid": self.process.pid}
            await self._stop()
            if not python.is_file():
                raise ValueError("Install the training engine before opening TensorBoard")
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            log_path = folder / "tensorboard.log"
            with log_path.open("ab") as log:
                self.process = await asyncio.create_subprocess_exec(str(python), "-m", "tensorboard.main",
                    "--logdir", str(folder), "--host", "127.0.0.1", "--port", str(port), "--load_fast=false",
                    cwd=str(cwd), env={**env, "CUDA_VISIBLE_DEVICES": ""}, stdout=log, stderr=log,
                    **({"creationflags": 0x08000000} if os.name == "nt" else {}))
            self.run_id, self.url = run_id, f"http://127.0.0.1:{port}"
            try:
                async with httpx.AsyncClient(timeout=1, trust_env=False) as client:
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        if self.process.returncode is not None:
                            raise ValueError("TensorBoard ended; see tensorboard.log in the run folder")
                        try:
                            response = await client.get(self.url)
                            if response.status_code == 200:
                                return {"url": self.url, "pid": self.process.pid}
                        except httpx.HTTPError:
                            pass
                        await asyncio.sleep(.25)
                raise ValueError("TensorBoard did not start within 30 seconds; see tensorboard.log")
            except BaseException:
                await self._stop()
                raise
