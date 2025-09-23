"""Comprehensive unit tests for error handling functionality."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from illustrator.error_handling import (
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    RecoveryAction,
    ErrorAnalyzer,
    RecoveryStrategy,
    ErrorRecoveryHandler,
    resilient_async,
    error_monitoring_context,
    HealthCheck,
    global_error_handler,
    safe_execute,
    get_global_error_stats
)


class TestErrorEnums:
    """Test error enumeration classes."""

    def test_error_severity_enum(self):
        """Test error severity levels."""
        assert ErrorSeverity.LOW == "low"
        assert ErrorSeverity.MEDIUM == "medium"
        assert ErrorSeverity.HIGH == "high"
        assert ErrorSeverity.CRITICAL == "critical"

    def test_error_category_enum(self):
        """Test error category types."""
        assert ErrorCategory.API_ERROR == "api_error"
        assert ErrorCategory.RATE_LIMIT == "rate_limit"
        assert ErrorCategory.TIMEOUT == "timeout"
        assert ErrorCategory.NETWORK_ERROR == "network_error"
        assert ErrorCategory.AUTHENTICATION_ERROR == "authentication_error"


class TestErrorContext:
    """Test error context data class."""

    def test_error_context_creation(self):
        """Test creating error context."""
        context = ErrorContext(
            function_name="test_function",
            input_data={"arg1": "value1"},
            attempt_number=2,
            max_attempts=3,
            error_category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            timestamp=time.time()
        )

        assert context.function_name == "test_function"
        assert context.attempt_number == 2
        assert context.error_category == ErrorCategory.TIMEOUT
        assert context.severity == ErrorSeverity.MEDIUM
        assert context.additional_info == {}

    def test_error_context_with_additional_info(self):
        """Test error context with additional information."""
        additional_info = {"provider": "openai", "model": "gpt-4"}
        context = ErrorContext(
            function_name="test_function",
            input_data={},
            attempt_number=1,
            max_attempts=3,
            error_category=ErrorCategory.API_ERROR,
            severity=ErrorSeverity.HIGH,
            timestamp=time.time(),
            additional_info=additional_info
        )

        assert context.additional_info == additional_info


class TestRecoveryAction:
    """Test recovery action data class."""

    def test_recovery_action_creation(self):
        """Test creating recovery action."""
        action = RecoveryAction(
            action_type="retry",
            delay_seconds=5.0,
            fallback_function=None
        )

        assert action.action_type == "retry"
        assert action.delay_seconds == 5.0
        assert action.fallback_kwargs == {}

    def test_recovery_action_with_fallback(self):
        """Test recovery action with fallback function."""
        def fallback_func():
            return "fallback_result"

        action = RecoveryAction(
            action_type="fallback",
            fallback_function=fallback_func,
            fallback_args=("arg1",),
            fallback_kwargs={"key": "value"}
        )

        assert action.action_type == "fallback"
        assert action.fallback_function == fallback_func
        assert action.fallback_args == ("arg1",)
        assert action.fallback_kwargs == {"key": "value"}


class TestErrorAnalyzer:
    """Test error analysis functionality."""

    def test_categorize_rate_limit_error(self):
        """Test rate limit error categorization."""
        error = Exception("Rate limit exceeded")
        category = ErrorAnalyzer.categorize_error(error)
        assert category == ErrorCategory.RATE_LIMIT

        error = Exception("Too many requests")
        category = ErrorAnalyzer.categorize_error(error)
        assert category == ErrorCategory.RATE_LIMIT

    def test_categorize_timeout_error(self):
        """Test timeout error categorization."""
        error = asyncio.TimeoutError("Request timed out")
        category = ErrorAnalyzer.categorize_error(error)
        assert category == ErrorCategory.TIMEOUT

        error = Exception("Connection timeout")
        category = ErrorAnalyzer.categorize_error(error)
        assert category == ErrorCategory.TIMEOUT

    def test_categorize_network_error(self):
        """Test network error categorization."""
        error = ConnectionError("Connection refused")
        category = ErrorAnalyzer.categorize_error(error)
        assert category == ErrorCategory.NETWORK_ERROR

        error = Exception("Network error occurred")
        category = ErrorAnalyzer.categorize_error(error)
        assert category == ErrorCategory.NETWORK_ERROR

    def test_categorize_authentication_error(self):
        """Test authentication error categorization."""
        error = Exception("Unauthorized access")
        category = ErrorAnalyzer.categorize_error(error)
        assert category == ErrorCategory.AUTHENTICATION_ERROR

        error = Exception("Invalid API key")
        category = ErrorAnalyzer.categorize_error(error)
        assert category == ErrorCategory.AUTHENTICATION_ERROR

    def test_categorize_unknown_error(self):
        """Test unknown error categorization."""
        error = Exception("Some random error")
        category = ErrorAnalyzer.categorize_error(error)
        assert category == ErrorCategory.PROCESSING_ERROR

    def test_assess_severity_critical_errors(self):
        """Test severity assessment for critical errors."""
        severity = ErrorAnalyzer.assess_severity(ErrorCategory.AUTHENTICATION_ERROR, 1)
        assert severity == ErrorSeverity.CRITICAL

        severity = ErrorAnalyzer.assess_severity(ErrorCategory.QUOTA_EXCEEDED, 1)
        assert severity == ErrorSeverity.HIGH

    def test_assess_severity_escalation(self):
        """Test severity escalation with repeated attempts."""
        # Low severity should escalate to medium after 3 attempts
        severity = ErrorAnalyzer.assess_severity(ErrorCategory.PROCESSING_ERROR, 4)
        assert severity == ErrorSeverity.MEDIUM

        # Medium severity should escalate to high after 3 attempts
        severity = ErrorAnalyzer.assess_severity(ErrorCategory.TIMEOUT, 4)
        assert severity == ErrorSeverity.HIGH


class TestRecoveryStrategy:
    """Test recovery strategy functionality."""

    def test_critical_error_recovery(self):
        """Test recovery action for critical errors."""
        context = ErrorContext(
            function_name="test",
            input_data={},
            attempt_number=1,
            max_attempts=3,
            error_category=ErrorCategory.AUTHENTICATION_ERROR,
            severity=ErrorSeverity.CRITICAL,
            timestamp=time.time()
        )

        action = RecoveryStrategy.get_recovery_action(context)
        assert action.action_type == "abort"

    def test_rate_limit_recovery(self):
        """Test recovery action for rate limit errors."""
        context = ErrorContext(
            function_name="test",
            input_data={},
            attempt_number=2,
            max_attempts=3,
            error_category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            timestamp=time.time()
        )

        action = RecoveryStrategy.get_recovery_action(context)
        assert action.action_type == "retry"
        assert action.delay_seconds > 0

    def test_timeout_recovery(self):
        """Test recovery action for timeout errors."""
        context = ErrorContext(
            function_name="test",
            input_data={},
            attempt_number=1,
            max_attempts=3,
            error_category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.MEDIUM,
            timestamp=time.time()
        )

        action = RecoveryStrategy.get_recovery_action(context)
        assert action.action_type == "retry"
        assert action.delay_seconds > 0

    def test_fallback_recovery(self):
        """Test recovery with fallback functions."""
        def fallback_func():
            return "fallback_result"

        fallback_functions = {"fallback_model": fallback_func}

        context = ErrorContext(
            function_name="test",
            input_data={},
            attempt_number=1,
            max_attempts=3,
            error_category=ErrorCategory.MODEL_ERROR,
            severity=ErrorSeverity.HIGH,
            timestamp=time.time()
        )

        action = RecoveryStrategy.get_recovery_action(context, fallback_functions)
        assert action.action_type == "fallback"
        assert action.fallback_function == fallback_func

    def test_max_attempts_exceeded(self):
        """Test recovery when max attempts exceeded."""
        context = ErrorContext(
            function_name="test",
            input_data={},
            attempt_number=3,
            max_attempts=3,
            error_category=ErrorCategory.PROCESSING_ERROR,
            severity=ErrorSeverity.LOW,
            timestamp=time.time()
        )

        action = RecoveryStrategy.get_recovery_action(context)
        assert action.action_type == "abort"


class TestErrorRecoveryHandler:
    """Test error recovery handler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = ErrorRecoveryHandler(max_attempts=3)

    def test_handler_initialization(self):
        """Test handler initialization."""
        assert self.handler.max_attempts == 3
        assert self.handler.error_history == []
        assert self.handler.recovery_stats["total_errors"] == 0

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful function execution."""
        async def test_function():
            return "success"

        result = await self.handler.handle_with_recovery(test_function)
        assert result == "success"
        assert self.handler.recovery_stats["total_errors"] == 0

    @pytest.mark.asyncio
    async def test_retry_and_success(self):
        """Test retry mechanism with eventual success."""
        call_count = 0

        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"

        result = await self.handler.handle_with_recovery(test_function)
        assert result == "success"
        assert call_count == 3
        assert self.handler.recovery_stats["recovered_errors"] == 1

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test behavior when max retries are exceeded."""
        async def test_function():
            raise Exception("Persistent failure")

        with pytest.raises(Exception, match="Persistent failure"):
            await self.handler.handle_with_recovery(test_function)

        assert self.handler.recovery_stats["failed_recoveries"] == 1

    @pytest.mark.asyncio
    async def test_fallback_execution(self):
        """Test fallback function execution."""
        async def test_function():
            raise Exception("Model error")

        async def fallback_function():
            return "fallback_result"

        fallback_functions = {"fallback_model": fallback_function}

        result = await self.handler.handle_with_recovery(
            test_function,
            fallback_functions=fallback_functions
        )
        assert result == "fallback_result"

    @pytest.mark.asyncio
    async def test_skip_action(self):
        """Test skip action for unrecoverable errors."""
        async def test_function():
            raise ValueError("Validation error")

        result = await self.handler.handle_with_recovery(test_function)
        assert result is None

    def test_error_statistics(self):
        """Test error statistics collection."""
        # Simulate some errors
        self.handler.recovery_stats["total_errors"] = 10
        self.handler.recovery_stats["recovered_errors"] = 7
        self.handler.recovery_stats["failed_recoveries"] = 3
        self.handler.recovery_stats["category_counts"] = {
            ErrorCategory.TIMEOUT: 5,
            ErrorCategory.RATE_LIMIT: 3
        }

        stats = self.handler.get_error_statistics()

        assert stats["total_errors"] == 10
        assert stats["recovered_errors"] == 7
        assert stats["recovery_rate"] == 0.7
        assert stats["category_breakdown"][ErrorCategory.TIMEOUT] == 5


class TestResilientDecorator:
    """Test resilient async decorator."""

    @pytest.mark.asyncio
    async def test_resilient_decorator_success(self):
        """Test resilient decorator with successful function."""
        @resilient_async(max_attempts=3)
        async def test_function():
            return "success"

        result = await test_function()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_resilient_decorator_with_retry(self):
        """Test resilient decorator with retry."""
        call_count = 0

        @resilient_async(max_attempts=3)
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"

        result = await test_function()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_resilient_decorator_error_stats(self):
        """Test error statistics from decorated function."""
        @resilient_async(max_attempts=2)
        async def test_function():
            raise Exception("Always fails")

        with pytest.raises(Exception):
            await test_function()

        # Check that error handler is attached
        assert hasattr(test_function, 'error_handler')
        stats = test_function.error_handler.get_error_statistics()
        assert stats["total_errors"] > 0


class TestErrorMonitoringContext:
    """Test error monitoring context manager."""

    @pytest.mark.asyncio
    async def test_monitoring_context_success(self):
        """Test monitoring context with successful operation."""
        async with error_monitoring_context("test_operation") as errors:
            # Simulate successful operation
            pass

        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_monitoring_context_with_error(self):
        """Test monitoring context with error."""
        with pytest.raises(ValueError):
            async with error_monitoring_context("test_operation") as errors:
                raise ValueError("Test error")

        # Note: errors list is populated but we can't access it after exception


class TestHealthCheck:
    """Test health check functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.health_check = HealthCheck()

    @pytest.mark.asyncio
    async def test_api_health_check_success(self):
        """Test successful API health check."""
        async def healthy_api():
            return "OK"

        health_status = await self.health_check.check_api_health("test_api", healthy_api)

        assert health_status["status"] == "healthy"
        assert health_status["error"] is None
        assert health_status["response_time"] > 0

    @pytest.mark.asyncio
    async def test_api_health_check_failure(self):
        """Test failed API health check."""
        async def unhealthy_api():
            raise Exception("API Error")

        health_status = await self.health_check.check_api_health("test_api", unhealthy_api)

        assert health_status["status"] == "unhealthy"
        assert "API Error" in health_status["error"]
        assert "error_category" in health_status

    @pytest.mark.asyncio
    async def test_comprehensive_health_check(self):
        """Test comprehensive health check across multiple APIs."""
        async def healthy_api():
            return "OK"

        async def unhealthy_api():
            raise Exception("API Error")

        api_tests = {
            "api1": healthy_api,
            "api2": healthy_api,
            "api3": unhealthy_api
        }

        overall_health = await self.health_check.comprehensive_health_check(api_tests)

        assert overall_health["total_apis"] == 3
        assert overall_health["healthy_apis"] == 2
        assert overall_health["overall_status"] == "degraded"
        assert overall_health["health_percentage"] == 2/3 * 100

    def test_health_summary(self):
        """Test health summary generation."""
        summary = self.health_check.get_health_summary()

        assert "last_check_time" in summary
        assert "api_status" in summary
        assert "system_uptime" in summary


class TestGlobalErrorHandler:
    """Test global error handling functions."""

    @pytest.mark.asyncio
    async def test_safe_execute(self):
        """Test safe execute function."""
        async def test_function():
            return "success"

        result = await safe_execute(test_function)
        assert result == "success"

    def test_global_error_stats(self):
        """Test global error statistics."""
        stats = get_global_error_stats()
        assert isinstance(stats, dict)
        assert "total_errors" in stats