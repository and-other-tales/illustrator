"""Comprehensive unit tests for parallel processing functionality."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from illustrator.parallel_processor import (
    ParallelProcessor,
    ProcessingResult,
    BatchConfig,
    ProcessingStats,
    RateLimitConfig,
    CircuitBreakerConfig,
    CircuitBreakerState
)


class TestProcessingResult:
    """Test processing result data class."""

    def test_processing_result_success(self):
        """Test successful processing result."""
        result = ProcessingResult(
            item_id="test-1",
            success=True,
            result="processed_data",
            processing_time=1.5
        )

        assert result.item_id == "test-1"
        assert result.success is True
        assert result.result == "processed_data"
        assert result.processing_time == 1.5
        assert result.error is None

    def test_processing_result_failure(self):
        """Test failed processing result."""
        error = Exception("Processing failed")
        result = ProcessingResult(
            item_id="test-1",
            success=False,
            error=error,
            processing_time=0.5
        )

        assert result.item_id == "test-1"
        assert result.success is False
        assert result.result is None
        assert result.error == error


class TestBatchConfig:
    """Test batch configuration."""

    def test_batch_config_defaults(self):
        """Test batch configuration with default values."""
        config = BatchConfig()

        assert config.batch_size == 5
        assert config.max_concurrent == 3
        assert config.delay_between_batches == 1.0
        assert config.timeout_per_item == 30.0

    def test_batch_config_custom(self):
        """Test batch configuration with custom values."""
        config = BatchConfig(
            batch_size=10,
            max_concurrent=5,
            delay_between_batches=2.0,
            timeout_per_item=60.0
        )

        assert config.batch_size == 10
        assert config.max_concurrent == 5
        assert config.delay_between_batches == 2.0
        assert config.timeout_per_item == 60.0


class TestRateLimitConfig:
    """Test rate limit configuration."""

    def test_rate_limit_config(self):
        """Test rate limit configuration creation."""
        config = RateLimitConfig(
            requests_per_minute=60,
            requests_per_hour=3600,
            burst_limit=10
        )

        assert config.requests_per_minute == 60
        assert config.requests_per_hour == 3600
        assert config.burst_limit == 10


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""

    def test_circuit_breaker_config(self):
        """Test circuit breaker configuration creation."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=30.0,
            half_open_max_calls=3
        )

        assert config.failure_threshold == 5
        assert config.recovery_timeout == 30.0
        assert config.half_open_max_calls == 3


class TestProcessingStats:
    """Test processing statistics."""

    def test_processing_stats_creation(self):
        """Test processing statistics creation."""
        stats = ProcessingStats()

        assert stats.total_items == 0
        assert stats.successful_items == 0
        assert stats.failed_items == 0
        assert stats.total_processing_time == 0.0
        assert stats.start_time > 0

    def test_processing_stats_calculations(self):
        """Test processing statistics calculations."""
        stats = ProcessingStats()
        stats.total_items = 10
        stats.successful_items = 8
        stats.failed_items = 2
        stats.total_processing_time = 15.0

        assert stats.success_rate == 0.8
        assert stats.failure_rate == 0.2
        assert stats.average_processing_time == 1.5


class TestParallelProcessor:
    """Test the main parallel processor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ParallelProcessor()

    def test_initialization(self):
        """Test processor initialization."""
        assert isinstance(self.processor.batch_config, BatchConfig)
        assert isinstance(self.processor.rate_limits, dict)
        assert isinstance(self.processor.circuit_breakers, dict)
        assert isinstance(self.processor.stats, ProcessingStats)

    def test_add_rate_limit(self):
        """Test adding rate limit configuration."""
        config = RateLimitConfig(requests_per_minute=100)
        self.processor.add_rate_limit("test_provider", config)

        assert "test_provider" in self.processor.rate_limits
        assert self.processor.rate_limits["test_provider"] == config

    def test_add_circuit_breaker(self):
        """Test adding circuit breaker configuration."""
        config = CircuitBreakerConfig(failure_threshold=3)
        self.processor.add_circuit_breaker("test_provider", config)

        assert "test_provider" in self.processor.circuit_breakers
        assert self.processor.circuit_breakers["test_provider"].config == config
        assert self.processor.circuit_breakers["test_provider"].state == CircuitBreakerState.CLOSED

    def test_check_rate_limit_allowed(self):
        """Test rate limit checking when allowed."""
        config = RateLimitConfig(requests_per_minute=60)
        self.processor.add_rate_limit("test_provider", config)

        # First request should be allowed
        allowed = self.processor._check_rate_limit("test_provider")
        assert allowed is True

    def test_check_rate_limit_exceeded(self):
        """Test rate limit checking when exceeded."""
        config = RateLimitConfig(requests_per_minute=1)  # Very low limit
        self.processor.add_rate_limit("test_provider", config)

        # Fill up the rate limit
        for _ in range(2):
            self.processor._check_rate_limit("test_provider")

        # Next request should be rate limited
        allowed = self.processor._check_rate_limit("test_provider")
        assert allowed is False

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        config = CircuitBreakerConfig(failure_threshold=3)
        self.processor.add_circuit_breaker("test_provider", config)

        allowed = self.processor._check_circuit_breaker("test_provider")
        assert allowed is True

    def test_circuit_breaker_failure_tracking(self):
        """Test circuit breaker failure tracking."""
        config = CircuitBreakerConfig(failure_threshold=2)
        self.processor.add_circuit_breaker("test_provider", config)

        # Record failures
        self.processor._record_circuit_breaker_result("test_provider", False)
        self.processor._record_circuit_breaker_result("test_provider", False)

        # Circuit should now be open
        breaker = self.processor.circuit_breakers["test_provider"]
        assert breaker.state == CircuitBreakerState.OPEN

    def test_circuit_breaker_success_reset(self):
        """Test circuit breaker reset on success."""
        config = CircuitBreakerConfig(failure_threshold=3)
        self.processor.add_circuit_breaker("test_provider", config)

        # Record some failures
        self.processor._record_circuit_breaker_result("test_provider", False)
        self.processor._record_circuit_breaker_result("test_provider", False)

        # Record success - should reset failure count
        self.processor._record_circuit_breaker_result("test_provider", True)

        breaker = self.processor.circuit_breakers["test_provider"]
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_process_single_item_success(self):
        """Test processing single item successfully."""
        async def process_func(item):
            return f"processed_{item}"

        result = await self.processor._process_single_item("test_item", process_func, "default")

        assert result.success is True
        assert result.result == "processed_test_item"
        assert result.item_id == "test_item"

    @pytest.mark.asyncio
    async def test_process_single_item_failure(self):
        """Test processing single item with failure."""
        async def process_func(item):
            raise Exception("Processing failed")

        result = await self.processor._process_single_item("test_item", process_func, "default")

        assert result.success is False
        assert "Processing failed" in str(result.error)
        assert result.item_id == "test_item"

    @pytest.mark.asyncio
    async def test_process_single_item_timeout(self):
        """Test processing single item with timeout."""
        async def slow_process_func(item):
            await asyncio.sleep(2)  # Longer than default timeout
            return f"processed_{item}"

        # Set short timeout
        self.processor.batch_config.timeout_per_item = 0.5

        result = await self.processor._process_single_item("test_item", slow_process_func, "default")

        assert result.success is False
        assert "timeout" in str(result.error).lower()

    @pytest.mark.asyncio
    async def test_process_single_item_rate_limited(self):
        """Test processing with rate limiting."""
        config = RateLimitConfig(requests_per_minute=1)
        self.processor.add_rate_limit("limited_provider", config)

        async def process_func(item):
            return f"processed_{item}"

        # First call should succeed
        result1 = await self.processor._process_single_item("item1", process_func, "limited_provider")
        assert result1.success is True

        # Second call should be rate limited
        result2 = await self.processor._process_single_item("item2", process_func, "limited_provider")
        assert result2.success is False
        assert "rate limit" in str(result2.error).lower()

    @pytest.mark.asyncio
    async def test_process_single_item_circuit_breaker_open(self):
        """Test processing with open circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=1)
        self.processor.add_circuit_breaker("failing_provider", config)

        # Trigger circuit breaker
        self.processor._record_circuit_breaker_result("failing_provider", False)

        async def process_func(item):
            return f"processed_{item}"

        result = await self.processor._process_single_item("test_item", process_func, "failing_provider")

        assert result.success is False
        assert "circuit breaker" in str(result.error).lower()

    @pytest.mark.asyncio
    async def test_process_batch_success(self):
        """Test processing batch successfully."""
        async def process_func(item):
            await asyncio.sleep(0.1)  # Simulate processing time
            return f"processed_{item}"

        items = ["item1", "item2", "item3"]
        results = await self.processor._process_batch(items, process_func, "default")

        assert len(results) == 3
        assert all(r.success for r in results)
        assert results[0].result == "processed_item1"

    @pytest.mark.asyncio
    async def test_process_batch_mixed_results(self):
        """Test processing batch with mixed success/failure."""
        async def process_func(item):
            if item == "fail_item":
                raise Exception("Intentional failure")
            return f"processed_{item}"

        items = ["item1", "fail_item", "item3"]
        results = await self.processor._process_batch(items, process_func, "default")

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True

    @pytest.mark.asyncio
    async def test_process_in_parallel(self):
        """Test complete parallel processing workflow."""
        async def process_func(item):
            await asyncio.sleep(0.1)
            return f"processed_{item}"

        items = ["item1", "item2", "item3", "item4", "item5"]

        # Configure small batches for testing
        self.processor.batch_config.batch_size = 2

        results = await self.processor.process_in_parallel(items, process_func)

        assert len(results) == 5
        successful_results = [r for r in results if r.success]
        assert len(successful_results) == 5

    @pytest.mark.asyncio
    async def test_process_in_parallel_with_provider(self):
        """Test parallel processing with specific provider."""
        config = RateLimitConfig(requests_per_minute=10)
        self.processor.add_rate_limit("test_provider", config)

        async def process_func(item):
            return f"processed_{item}"

        items = ["item1", "item2"]
        results = await self.processor.process_in_parallel(
            items, process_func, provider_name="test_provider"
        )

        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_process_in_parallel_empty_items(self):
        """Test parallel processing with empty item list."""
        async def process_func(item):
            return f"processed_{item}"

        results = await self.processor.process_in_parallel([], process_func)
        assert results == []

    def test_get_processing_stats(self):
        """Test getting processing statistics."""
        # Simulate some processing
        self.processor.stats.total_items = 10
        self.processor.stats.successful_items = 8
        self.processor.stats.failed_items = 2
        self.processor.stats.total_processing_time = 15.0

        stats = self.processor.get_processing_stats()

        assert stats.total_items == 10
        assert stats.successful_items == 8
        assert stats.failed_items == 2
        assert stats.success_rate == 0.8

    def test_reset_stats(self):
        """Test resetting processing statistics."""
        # Set some stats
        self.processor.stats.total_items = 10
        self.processor.stats.successful_items = 8

        self.processor.reset_stats()

        assert self.processor.stats.total_items == 0
        assert self.processor.stats.successful_items == 0

    def test_get_rate_limit_status(self):
        """Test getting rate limit status."""
        config = RateLimitConfig(requests_per_minute=60)
        self.processor.add_rate_limit("test_provider", config)

        # Make some requests
        self.processor._check_rate_limit("test_provider")
        self.processor._check_rate_limit("test_provider")

        status = self.processor.get_rate_limit_status("test_provider")

        assert "current_minute_count" in status
        assert "current_hour_count" in status
        assert "requests_per_minute" in status
        assert status["current_minute_count"] == 2

    def test_get_circuit_breaker_status(self):
        """Test getting circuit breaker status."""
        config = CircuitBreakerConfig(failure_threshold=5)
        self.processor.add_circuit_breaker("test_provider", config)

        # Record some failures
        self.processor._record_circuit_breaker_result("test_provider", False)
        self.processor._record_circuit_breaker_result("test_provider", False)

        status = self.processor.get_circuit_breaker_status("test_provider")

        assert status["state"] == CircuitBreakerState.CLOSED
        assert status["failure_count"] == 2
        assert status["failure_threshold"] == 5

    @pytest.mark.asyncio
    async def test_process_with_custom_batch_config(self):
        """Test processing with custom batch configuration."""
        custom_config = BatchConfig(
            batch_size=3,
            max_concurrent=2,
            delay_between_batches=0.1
        )

        processor = ParallelProcessor(batch_config=custom_config)

        async def process_func(item):
            return f"processed_{item}"

        items = ["item1", "item2", "item3", "item4", "item5"]
        results = await processor.process_in_parallel(items, process_func)

        assert len(results) == 5
        assert all(r.success for r in results)

    def test_update_batch_config(self):
        """Test updating batch configuration."""
        new_config = BatchConfig(batch_size=10, max_concurrent=5)
        self.processor.update_batch_config(new_config)

        assert self.processor.batch_config.batch_size == 10
        assert self.processor.batch_config.max_concurrent == 5