"""Parallel processing optimizations for manuscript illustration generation."""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Coroutine, TypeVar, Generic
from functools import wraps
import time
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class ProcessingTask(Generic[T, R]):
    """Represents a processing task with metadata."""
    task_id: str
    input_data: T
    processor_func: Callable[[T], Coroutine[Any, Any, R]]
    priority: int = 0  # Higher numbers = higher priority
    estimated_duration: float = 1.0  # seconds
    dependencies: List[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 300.0  # 5 minutes default

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ProcessingResult(Generic[R]):
    """Result of a processing task."""
    task_id: str
    success: bool
    result: Optional[R]
    error: Optional[str]
    duration: float
    retry_count: int
    start_time: float
    end_time: float


class TaskQueue(Generic[T, R]):
    """Thread-safe priority task queue with dependency management."""

    def __init__(self):
        self._queue = []
        self._lock = threading.Lock()
        self._completed_tasks: set = set()
        self._in_progress_tasks: set = set()

    def add_task(self, task: ProcessingTask[T, R]):
        """Add a task to the queue."""
        with self._lock:
            self._queue.append(task)
            # Sort by priority (higher priority first)
            self._queue.sort(key=lambda t: t.priority, reverse=True)

    def get_ready_task(self) -> Optional[ProcessingTask[T, R]]:
        """Get the next task that has all dependencies completed."""
        with self._lock:
            for i, task in enumerate(self._queue):
                if (task.task_id not in self._in_progress_tasks and
                    all(dep in self._completed_tasks for dep in task.dependencies)):
                    self._queue.pop(i)
                    self._in_progress_tasks.add(task.task_id)
                    return task
            return None

    def mark_completed(self, task_id: str):
        """Mark a task as completed."""
        with self._lock:
            self._completed_tasks.add(task_id)
            self._in_progress_tasks.discard(task_id)

    def mark_failed(self, task_id: str):
        """Mark a task as failed (removes from in-progress)."""
        with self._lock:
            self._in_progress_tasks.discard(task_id)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._queue) == 0

    def pending_count(self) -> int:
        """Get count of pending tasks."""
        with self._lock:
            return len(self._queue)


class RateLimiter:
    """Rate limiter for API calls with different providers."""

    def __init__(self):
        self._limits: Dict[str, Dict[str, Any]] = {
            'dalle': {'calls_per_minute': 50, 'calls_per_hour': 500},
            'imagen4': {'calls_per_minute': 100, 'calls_per_hour': 1000},
            'flux': {'calls_per_minute': 200, 'calls_per_hour': 2000},
            'llm_analysis': {'calls_per_minute': 500, 'calls_per_hour': 5000}
        }

        self._call_history: Dict[str, deque] = defaultdict(deque)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def acquire(self, provider: str) -> bool:
        """Acquire permission to make a call to the provider."""
        if provider not in self._limits:
            return True  # No limits for unknown providers

        async with self._locks[provider]:
            now = time.time()
            history = self._call_history[provider]
            limits = self._limits[provider]

            # Remove old entries
            while history and now - history[0] > 3600:  # 1 hour
                history.popleft()

            # Check hourly limit
            if len(history) >= limits['calls_per_hour']:
                return False

            # Check per-minute limit
            recent_calls = sum(1 for call_time in history if now - call_time < 60)
            if recent_calls >= limits['calls_per_minute']:
                return False

            # Record this call
            history.append(now)
            return True

    async def wait_for_permission(self, provider: str, max_wait: float = 300.0):
        """Wait until we have permission to make a call."""
        start_time = time.time()

        while time.time() - start_time < max_wait:
            if await self.acquire(provider):
                return True

            # Calculate wait time
            async with self._locks[provider]:
                history = self._call_history[provider]
                limits = self._limits[provider]

                if not history:
                    return True

                now = time.time()

                # Check what's limiting us
                recent_calls = [t for t in history if now - t < 60]
                hourly_calls = len(history)

                if hourly_calls >= limits['calls_per_hour']:
                    # Wait until oldest call is > 1 hour old
                    wait_time = 3600 - (now - history[0]) + 1
                elif len(recent_calls) >= limits['calls_per_minute']:
                    # Wait until oldest recent call is > 1 minute old
                    wait_time = 60 - (now - recent_calls[0]) + 1
                else:
                    wait_time = 1

                wait_time = min(wait_time, 30)  # Cap at 30 seconds

            await asyncio.sleep(wait_time)

        return False


class CircuitBreaker:
    """Circuit breaker pattern for resilient API calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._failure_count = 0
        self._last_failure_time = None
        self._state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs):
        """Execute a function with circuit breaker protection."""
        async with self._lock:
            if self._state == 'OPEN':
                if self._should_attempt_reset():
                    self._state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result

        except self.expected_exception as e:
            await self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        return (
            self._last_failure_time and
            time.time() - self._last_failure_time >= self.recovery_timeout
        )

    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            self._failure_count = 0
            self._state = 'CLOSED'

    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = 'OPEN'


class ParallelProcessor:
    """High-performance parallel processor for manuscript illustration tasks."""

    def __init__(
        self,
        max_concurrent_llm: int = 10,
        max_concurrent_image: int = 5,
        max_workers: int = None,
        enable_rate_limiting: bool = True,
        enable_circuit_breaker: bool = True
    ):
        self.max_concurrent_llm = max_concurrent_llm
        self.max_concurrent_image = max_concurrent_image
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)

        # Semaphores for controlling concurrency
        self.llm_semaphore = asyncio.Semaphore(max_concurrent_llm)
        self.image_semaphore = asyncio.Semaphore(max_concurrent_image)

        # Rate limiting and circuit breakers
        self.rate_limiter = RateLimiter() if enable_rate_limiting else None
        self.circuit_breakers = {} if enable_circuit_breaker else None

        # Task management
        self.task_queues: Dict[str, TaskQueue] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}

        # Performance tracking
        self.performance_stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0.0,
            'average_task_time': 0.0,
            'api_calls': defaultdict(int),
            'rate_limit_waits': 0,
            'circuit_breaker_opens': 0
        }

    async def process_chapters_parallel(
        self,
        chapters: List[Any],
        analysis_func: Callable,
        illustration_func: Callable,
        max_concurrent_chapters: int = 3
    ) -> List[Dict[str, Any]]:
        """Process multiple chapters in parallel."""

        chapter_semaphore = asyncio.Semaphore(max_concurrent_chapters)
        results = []

        async def process_single_chapter(chapter, chapter_index):
            async with chapter_semaphore:
                try:
                    logger.info(f"Starting parallel processing of chapter {chapter.number}")

                    # Analyze chapter for emotional moments
                    analysis_start = time.time()
                    emotional_moments = await self._rate_limited_call(
                        'llm_analysis',
                        analysis_func,
                        chapter
                    )
                    analysis_time = time.time() - analysis_start

                    # Process illustration generation in parallel batches
                    illustration_results = await self._process_illustrations_batch(
                        emotional_moments,
                        illustration_func,
                        chapter,
                        batch_size=3
                    )

                    return {
                        'chapter': chapter,
                        'chapter_index': chapter_index,
                        'emotional_moments': emotional_moments,
                        'illustrations': illustration_results,
                        'analysis_time': analysis_time,
                        'success': True
                    }

                except Exception as e:
                    logger.error(f"Error processing chapter {chapter.number}: {e}")
                    return {
                        'chapter': chapter,
                        'chapter_index': chapter_index,
                        'error': str(e),
                        'success': False
                    }

        # Create tasks for all chapters
        tasks = [
            asyncio.create_task(process_single_chapter(chapter, i))
            for i, chapter in enumerate(chapters)
        ]

        # Wait for all chapters to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Chapter processing failed with exception: {result}")
                processed_results.append({
                    'success': False,
                    'error': str(result)
                })
            else:
                processed_results.append(result)

        return processed_results

    async def _process_illustrations_batch(
        self,
        emotional_moments: List[Any],
        illustration_func: Callable,
        chapter: Any,
        batch_size: int = 3
    ) -> List[Dict[str, Any]]:
        """Process illustrations in parallel batches to optimize API usage."""

        all_results = []

        # Process in batches to avoid overwhelming APIs
        for i in range(0, len(emotional_moments), batch_size):
            batch = emotional_moments[i:i + batch_size]

            batch_tasks = []
            for j, moment in enumerate(batch):
                task = asyncio.create_task(
                    self._generate_single_illustration(
                        moment,
                        illustration_func,
                        chapter,
                        f"batch_{i//batch_size}_item_{j}"
                    )
                )
                batch_tasks.append(task)

            # Wait for current batch to complete before starting next
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Process batch results
            for k, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Illustration generation failed: {result}")
                    all_results.append({
                        'success': False,
                        'error': str(result),
                        'moment_index': i + k
                    })
                else:
                    all_results.append(result)

            # Brief pause between batches to be respectful to APIs
            if i + batch_size < len(emotional_moments):
                await asyncio.sleep(1)

        return all_results

    async def _generate_single_illustration(
        self,
        emotional_moment: Any,
        illustration_func: Callable,
        chapter: Any,
        task_id: str
    ) -> Dict[str, Any]:
        """Generate a single illustration with proper rate limiting and error handling."""

        async with self.image_semaphore:
            try:
                start_time = time.time()

                # Determine provider for rate limiting
                provider = getattr(illustration_func, '_provider', 'unknown')

                result = await self._rate_limited_call(
                    provider,
                    illustration_func,
                    emotional_moment,
                    chapter
                )

                duration = time.time() - start_time

                # Update performance stats
                self.performance_stats['tasks_completed'] += 1
                self.performance_stats['total_processing_time'] += duration
                self.performance_stats['average_task_time'] = (
                    self.performance_stats['total_processing_time'] /
                    self.performance_stats['tasks_completed']
                )

                return {
                    'success': True,
                    'result': result,
                    'duration': duration,
                    'task_id': task_id,
                    'emotional_moment': emotional_moment
                }

            except Exception as e:
                self.performance_stats['tasks_failed'] += 1
                logger.error(f"Illustration generation failed for {task_id}: {e}")

                return {
                    'success': False,
                    'error': str(e),
                    'task_id': task_id,
                    'emotional_moment': emotional_moment
                }

    async def _rate_limited_call(
        self,
        provider: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Make a rate-limited API call with circuit breaker protection."""

        # Rate limiting
        if self.rate_limiter:
            if not await self.rate_limiter.wait_for_permission(provider):
                self.performance_stats['rate_limit_waits'] += 1
                raise Exception(f"Rate limit exceeded for {provider}")

        # Circuit breaker protection
        if self.circuit_breakers:
            if provider not in self.circuit_breakers:
                self.circuit_breakers[provider] = CircuitBreaker()

            try:
                return await self.circuit_breakers[provider].call(func, *args, **kwargs)
            except Exception as e:
                if self.circuit_breakers[provider]._state == 'OPEN':
                    self.performance_stats['circuit_breaker_opens'] += 1
                raise e
        else:
            return await func(*args, **kwargs)

    async def process_with_dependencies(
        self,
        tasks: List[ProcessingTask],
        progress_callback: Optional[Callable] = None
    ) -> List[ProcessingResult]:
        """Process tasks with dependency management."""

        # Create task queue
        task_queue = TaskQueue()
        for task in tasks:
            task_queue.add_task(task)

        results = []
        completed_count = 0
        total_count = len(tasks)

        # Process tasks until queue is empty
        while not task_queue.is_empty() or len(self.active_tasks) > 0:

            # Start new tasks up to concurrency limit
            while len(self.active_tasks) < self.max_workers and not task_queue.is_empty():
                ready_task = task_queue.get_ready_task()
                if ready_task:
                    async_task = asyncio.create_task(
                        self._execute_task(ready_task)
                    )
                    self.active_tasks[ready_task.task_id] = async_task
                else:
                    break  # No ready tasks available

            if not self.active_tasks:
                break  # No tasks running and none ready (possible deadlock)

            # Wait for at least one task to complete
            done, pending = await asyncio.wait(
                self.active_tasks.values(),
                return_when=asyncio.FIRST_COMPLETED
            )

            # Process completed tasks
            for completed_task in done:
                result = await completed_task
                results.append(result)

                if result.success:
                    task_queue.mark_completed(result.task_id)
                else:
                    task_queue.mark_failed(result.task_id)

                # Remove from active tasks
                if result.task_id in self.active_tasks:
                    del self.active_tasks[result.task_id]

                completed_count += 1

                # Progress callback
                if progress_callback:
                    await progress_callback(completed_count, total_count, result)

        return results

    async def _execute_task(self, task: ProcessingTask) -> ProcessingResult:
        """Execute a single task with retry logic."""

        start_time = time.time()

        for attempt in range(task.max_retries + 1):
            try:
                # Apply timeout
                result = await asyncio.wait_for(
                    task.processor_func(task.input_data),
                    timeout=task.timeout
                )

                end_time = time.time()

                return ProcessingResult(
                    task_id=task.task_id,
                    success=True,
                    result=result,
                    error=None,
                    duration=end_time - start_time,
                    retry_count=attempt,
                    start_time=start_time,
                    end_time=end_time
                )

            except asyncio.TimeoutError:
                error_msg = f"Task {task.task_id} timed out after {task.timeout} seconds"
                logger.warning(f"{error_msg} (attempt {attempt + 1})")

                if attempt == task.max_retries:
                    return ProcessingResult(
                        task_id=task.task_id,
                        success=False,
                        result=None,
                        error=error_msg,
                        duration=time.time() - start_time,
                        retry_count=attempt,
                        start_time=start_time,
                        end_time=time.time()
                    )

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Task {task.task_id} failed: {error_msg} (attempt {attempt + 1})")

                if attempt == task.max_retries:
                    return ProcessingResult(
                        task_id=task.task_id,
                        success=False,
                        result=None,
                        error=error_msg,
                        duration=time.time() - start_time,
                        retry_count=attempt,
                        start_time=start_time,
                        end_time=time.time()
                    )

                # Exponential backoff for retries
                wait_time = min(2 ** attempt, 30)
                await asyncio.sleep(wait_time)

        # Should never reach here
        return ProcessingResult(
            task_id=task.task_id,
            success=False,
            result=None,
            error="Unexpected end of retry loop",
            duration=time.time() - start_time,
            retry_count=task.max_retries,
            start_time=start_time,
            end_time=time.time()
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.performance_stats.copy()
        stats['active_tasks'] = len(self.active_tasks)
        stats['success_rate'] = (
            stats['tasks_completed'] /
            max(1, stats['tasks_completed'] + stats['tasks_failed'])
        )
        return stats

    async def shutdown(self):
        """Gracefully shutdown the processor."""
        # Cancel all active tasks
        for task in self.active_tasks.values():
            task.cancel()

        # Wait for cancellation to complete
        if self.active_tasks:
            await asyncio.gather(
                *self.active_tasks.values(),
                return_exceptions=True
            )

        logger.info("ParallelProcessor shutdown complete")


def parallel_processor_decorator(
    provider: str = 'unknown',
    max_retries: int = 3,
    timeout: float = 300.0
):
    """Decorator to mark functions for parallel processing."""
    def decorator(func):
        func._provider = provider
        func._max_retries = max_retries
        func._timeout = timeout

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        return wrapper
    return decorator


# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor and log performance metrics."""

    def __init__(self, log_interval: float = 60.0):
        self.log_interval = log_interval
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.metrics_history = []

    async def start_monitoring(self, processor: ParallelProcessor):
        """Start performance monitoring loop."""
        while True:
            await asyncio.sleep(self.log_interval)

            current_time = time.time()
            stats = processor.get_performance_stats()

            # Add timestamp and duration
            stats['timestamp'] = current_time
            stats['elapsed_time'] = current_time - self.start_time

            self.metrics_history.append(stats)

            # Log current performance
            logger.info(
                f"Performance: "
                f"Tasks completed: {stats['tasks_completed']}, "
                f"Failed: {stats['tasks_failed']}, "
                f"Success rate: {stats['success_rate']:.2%}, "
                f"Avg task time: {stats['average_task_time']:.2f}s, "
                f"Active tasks: {stats['active_tasks']}"
            )

            # Keep only last 100 entries
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]