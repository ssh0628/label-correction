"""Release DataLoader workers without masking the original training exception."""
from __future__ import annotations

import logging
import sys


def close_loader(loader) -> None:
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    primary_error = sys.exc_info()[1]
    try:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
    except BaseException as shutdown_error:
        try:
            _force_stop_workers(iterator)
        except Exception:
            logging.getLogger(__name__).error("Forced DataLoader cleanup also failed.", exc_info=True)
        if primary_error is None or not isinstance(shutdown_error, Exception):
            raise
        logging.getLogger(__name__).warning(
            "DataLoader cleanup also failed; preserving the original training/evaluation exception.",
            exc_info=True,
        )
    finally:
        loader._iterator = None


def _force_stop_workers(iterator) -> None:
    """Stop only this iterator's children if normal PyTorch shutdown fails."""
    workers = getattr(iterator, "_workers", ())
    for worker in workers:
        try:
            if worker.is_alive():
                worker.terminate()
        except (OSError, ValueError, AssertionError, RuntimeError):
            pass
    for worker in workers:
        try:
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=1.0)
            if worker.is_alive():
                logging.getLogger(__name__).error("DataLoader worker %s did not exit after kill.", worker.pid)
        except (OSError, ValueError, AssertionError, RuntimeError):
            logging.getLogger(__name__).warning("Could not finish DataLoader worker cleanup.", exc_info=True)
    for queue in getattr(iterator, "_index_queues", ()):
        try:
            queue.cancel_join_thread()
            queue.close()
        except (OSError, ValueError, RuntimeError):
            pass
