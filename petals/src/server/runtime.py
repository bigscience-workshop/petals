import multiprocessing as mp
import multiprocessing.pool
import threading
from collections import defaultdict
from itertools import chain
from queue import SimpleQueue
from selectors import EVENT_READ, DefaultSelector
from statistics import mean
from time import time
from typing import Dict, NamedTuple, Optional

import torch
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.utils import get_logger
from prefetch_generator import BackgroundGenerator

logger = get_logger(__name__)


class Runtime(threading.Thread):
    """
    A group of processes that processes incoming requests for multiple module backends on a shared device.
    Runtime is usually created and managed by Server, humans need not apply.

    For debugging, you can start runtime manually with .start() or .run()

    >>> module_backends = {'block_uid': ModuleBackend(**kwargs)}
    >>> runtime = Runtime(module_backends)
    >>> runtime.start()  # start runtime in background thread. To start in current thread, use runtime.run()
    >>> runtime.ready.wait()  # await for runtime to load all blocks on device and create request pools
    >>> future = runtime.module_backends['block_uid'].forward_pool.submit_task(*module_inputs)
    >>> print("Returned:", future.result())
    >>> runtime.shutdown()

    :param module_backends: a dict [block uid -> ModuleBackend]
    :param prefetch_batches: form up to this many batches in advance
    :param sender_threads: dispatches outputs from finished batches using this many asynchronous threads
    :param device: if specified, moves all blocks and data to this device via .to(device=device).
      If you want to manually specify devices for each block (in their forward pass), leave device=None (default)

    :param stats_report_interval: interval to collect and log statistics about runtime performance
    """

    SHUTDOWN_TRIGGER = "RUNTIME SHUTDOWN TRIGGERED"

    def __init__(
        self,
        module_backends: Dict[str, ModuleBackend],
        prefetch_batches: int = 1,
        sender_threads: int = 1,
        device: torch.device = None,
        stats_report_interval: Optional[int] = None,
    ):
        super().__init__()
        self.module_backends = module_backends
        self.pools = tuple(chain(*(backend.get_pools() for backend in module_backends.values())))
        self.device, self.prefetch_batches, self.sender_threads = device, prefetch_batches, sender_threads
        self.shutdown_recv, self.shutdown_send = mp.Pipe(duplex=False)
        self.shutdown_trigger = mp.Event()
        self.ready = mp.Event()  # event is set iff server is currently running and ready to accept batches

        self.stats_report_interval = stats_report_interval
        if self.stats_report_interval is not None:
            self.stats_reporter = StatsReporter(self.stats_report_interval)

    def run(self):
        for pool in self.pools:
            if not pool.is_alive():
                pool.start()
        if self.device is not None:
            for backend in self.module_backends.values():
                backend.module.to(self.device)

        with mp.pool.ThreadPool(self.sender_threads) as output_sender_pool:
            try:
                self.ready.set()
                if self.stats_report_interval is not None:
                    self.stats_reporter.start()
                logger.info("Started")

                batch_iterator = self.iterate_minibatches_from_pools()
                if self.prefetch_batches > 0:
                    batch_iterator = BackgroundGenerator(batch_iterator, self.prefetch_batches)

                for pool, batch_index, batch in batch_iterator:
                    logger.debug(f"Processing batch {batch_index} from pool {pool.name}")

                    start = time()
                    try:
                        outputs = pool.process_func(*batch)
                        output_sender_pool.apply_async(pool.send_outputs_from_runtime, args=[batch_index, outputs])

                        batch_processing_time = time() - start

                        batch_size = outputs[0].size(0)
                        logger.debug(f"Pool {pool.name}: batch {batch_index} processed, size {batch_size}")

                        if self.stats_report_interval is not None:
                            self.stats_reporter.report_stats(pool.name, batch_size, batch_processing_time)

                    except KeyboardInterrupt:
                        raise
                    except BaseException as exception:
                        logger.exception(f"Caught {exception}, attempting to recover")
                        output_sender_pool.apply_async(pool.send_exception_from_runtime, args=[batch_index, exception])

            finally:
                if not self.shutdown_trigger.is_set():
                    self.shutdown()

    def shutdown(self):
        """Gracefully terminate a running runtime."""
        logger.info("Shutting down")
        self.ready.clear()

        if self.stats_report_interval is not None:
            self.stats_reporter.stop.set()
            self.stats_reporter.join()

        logger.debug("Terminating pools")
        for pool in self.pools:
            if pool.is_alive():
                pool.shutdown()
        logger.debug("Pools terminated")

        # trigger background thread to shutdown
        self.shutdown_send.send(self.SHUTDOWN_TRIGGER)
        self.shutdown_trigger.set()

    def iterate_minibatches_from_pools(self, timeout=None):
        """
        Chooses pool according to priority, then copies exposed batch and frees the buffer
        """
        with DefaultSelector() as selector:
            for pool in self.pools:
                selector.register(pool.batch_receiver, EVENT_READ, pool)
            selector.register(self.shutdown_recv, EVENT_READ, self.SHUTDOWN_TRIGGER)

            while True:
                # wait until at least one batch_receiver becomes available
                logger.debug("Waiting for inputs from task pools")
                ready_fds = selector.select()
                ready_objects = {key.data for (key, events) in ready_fds}
                if self.SHUTDOWN_TRIGGER in ready_objects:
                    break  # someone asked us to shutdown, break from the loop

                logger.debug("Choosing the pool with first priority")

                pool = min(ready_objects, key=lambda pool: pool.priority)

                logger.debug(f"Loading batch from {pool.name}")
                batch_index, batch_tensors = pool.load_batch_to_runtime(timeout, self.device)
                logger.debug(f"Loaded batch from {pool.name}")
                yield pool, batch_index, batch_tensors


BatchStats = NamedTuple("BatchStats", (("batch_size", int), ("processing_time", float)))


class StatsReporter(threading.Thread):
    def __init__(self, report_interval: int):
        super().__init__()
        self.report_interval = report_interval
        self.stop = threading.Event()
        self.stats_queue = SimpleQueue()

    def run(self):
        while not self.stop.wait(self.report_interval):
            pool_batch_stats = defaultdict(list)
            while not self.stats_queue.empty():
                pool_uid, batch_stats = self.stats_queue.get()
                pool_batch_stats[pool_uid].append(batch_stats)

            total_processed_batches = sum(len(pool_stats) for pool_stats in pool_batch_stats.values())
            logger.info(f"Processed {total_processed_batches} batches in last {self.report_interval} seconds:")
            for pool_uid, pool_stats in pool_batch_stats.items():
                total_batches = len(pool_stats)
                total_examples = sum(batch_stats.batch_size for batch_stats in pool_stats)
                avg_batch_size = mean(batch_stats.batch_size for batch_stats in pool_stats)
                total_time = sum(batch_stats.processing_time for batch_stats in pool_stats)
                batches_to_time = total_batches / total_time
                batch_performance = f"{batches_to_time:.2f} " + ("batches/s" if batches_to_time > 1 else "s/batch")

                examples_to_time = total_examples / total_time
                example_performance = f"{examples_to_time:.2f} " + (
                    "examples/s" if examples_to_time > 1 else "s/example"
                )

                logger.info(
                    f"{pool_uid}: "
                    f"{total_batches} batches ({batch_performance}), "
                    f"{total_examples} examples ({example_performance}), "
                    f"avg batch size {avg_batch_size:.2f}"
                )

    def report_stats(self, pool_uid, batch_size, processing_time):
        batch_stats = BatchStats(batch_size, processing_time)
        self.stats_queue.put_nowait((pool_uid, batch_stats))
