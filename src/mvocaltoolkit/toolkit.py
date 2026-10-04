"""The toolkit object: settings, catalogs, models, engines and jobs in one place.

Used by the HTTP server and by the CLI (which runs operations in-process without a server).
"""

from __future__ import annotations

from pathlib import Path

from .jobs import JobManager
from .models.catalog import Catalog
from .models.store import ModelStore
from .engines.manager import EngineManager
from .settings import Home, Settings


class Toolkit:
    def __init__(self, home: Home | None = None, settings: Settings | None = None, engine_dirs: list[Path] | None = None):
        self.home = home or Home()
        self.home.root.mkdir(parents=True, exist_ok=True)
        self.settings = settings or self.home.load_settings()
        self.catalog = Catalog(self.home, self.settings)
        self.catalog.load()
        self.models = ModelStore(self.home, self.settings, self.catalog)
        self.engines = EngineManager(self.home, self.settings, extra_dirs=engine_dirs)
        self._jobs: JobManager | None = None

    @property
    def jobs(self) -> JobManager:
        # created lazily: needs a running event loop
        if self._jobs is None:
            self._jobs = JobManager(self.home.jobs, max_parallel=self.settings.max_parallel_jobs)
        return self._jobs

    async def start(self) -> None:
        self.engines.start_idle_watcher()

    async def stop(self) -> None:
        await self.engines.stop_all()
