"""Advanced error handling and resilience patterns for manuscript illustration generation."""

import asyncio
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
from enum import Enum
from functools import wraps
import time
import random
from contextlib import asynccontextmanager


logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error category types."""
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    PROCESSING_ERROR = "processing_error"
    VALIDATION_ERROR = "validation_error"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "authentication_error"
    QUOTA_EXCEEDED = "quota_exceeded"
    MODEL_ERROR = "model_error"
    DATA_ERROR = "data_error"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    function_name: str
    input_data: Any
    attempt_number: int
    max_attempts: int
    error_category: ErrorCategory
    severity: ErrorSeverity
    timestamp: float
    additional_info: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}


@dataclass
class RecoveryAction:
    """Defines a recovery action for error handling."""
    action_type: str  # retry, fallback, skip, abort
    delay_seconds: float = 0.0
    fallback_function: Optional[Callable] = None
    fallback_args: tuple = ()
    fallback_kwargs: Dict[str, Any] = None
    success_condition: Optional[Callable] = None

    def __post_init__(self):
        if self.fallback_kwargs is None:
            self.fallback_kwargs = {}


class ErrorAnalyzer:
    """Analyzes errors to determine appropriate recovery strategies."""

    # Error patterns and their categories
    ERROR_PATTERNS = {
        ErrorCategory.RATE_LIMIT: [
            'rate limit', 'too many requests', 'quota exceeded', 'requests per minute',
            'rate_limit_exceeded', 'throttled', '429'
        ],
        ErrorCategory.TIMEOUT: [
            'timeout', 'timed out', 'connection timeout', 'read timeout',
            'asyncio.TimeoutError', 'request timeout'
        ],
        ErrorCategory.NETWORK_ERROR: [
            'connection error', 'network error', 'connection refused',
            'connection reset', 'dns', 'socket', 'unreachable'
        ],
        ErrorCategory.AUTHENTICATION_ERROR: [
            'authentication', 'unauthorized', 'invalid api key', 'forbidden',
            '401', '403', 'access denied', 'invalid token'
        ],
        ErrorCategory.QUOTA_EXCEEDED: [
            'quota', 'limit exceeded', 'usage limit', 'billing',
            'insufficient funds', 'credits'
        ],
        ErrorCategory.MODEL_ERROR: [
            'model error', 'model not found', 'invalid model', 'model unavailable',
            'content policy', 'safety filter', 'inappropriate content'
        ]
    }

    @classmethod
    def categorize_error(cls, error: Exception, error_message: str = None) -> ErrorCategory:
        """Categorize an error based on its type and message."""
        error_text = (error_message or str(error)).lower()
        error_type = type(error).__name__.lower()

        # Check for specific error patterns
        for category, patterns in cls.ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern in error_text or pattern in error_type:
                    return category

        # Default categorization based on exception type
        if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
            return ErrorCategory.TIMEOUT
        elif isinstance(error, (ConnectionError, OSError)):
            return ErrorCategory.NETWORK_ERROR
        elif isinstance(error, ValueError):
            return ErrorCategory.VALIDATION_ERROR
        else:
            return ErrorCategory.PROCESSING_ERROR

    @classmethod
    def assess_severity(cls, error_category: ErrorCategory, attempt_number: int) -> ErrorSeverity:
        """Assess the severity of an error."""
        severity_mapping = {
            ErrorCategory.AUTHENTICATION_ERROR: ErrorSeverity.CRITICAL,
            ErrorCategory.QUOTA_EXCEEDED: ErrorSeverity.HIGH,
            ErrorCategory.MODEL_ERROR: ErrorSeverity.HIGH,
            ErrorCategory.RATE_LIMIT: ErrorSeverity.MEDIUM,
            ErrorCategory.TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorCategory.NETWORK_ERROR: ErrorSeverity.MEDIUM,
            ErrorCategory.PROCESSING_ERROR: ErrorSeverity.LOW,
            ErrorCategory.VALIDATION_ERROR: ErrorSeverity.LOW,
            ErrorCategory.DATA_ERROR: ErrorSeverity.LOW,
        }

        base_severity = severity_mapping.get(error_category, ErrorSeverity.MEDIUM)

        # Escalate severity with repeated attempts
        if attempt_number > 3:
            if base_severity == ErrorSeverity.LOW:
                return ErrorSeverity.MEDIUM
            elif base_severity == ErrorSeverity.MEDIUM:
                return ErrorSeverity.HIGH

        return base_severity


class RecoveryStrategy:
    """Defines recovery strategies for different error scenarios."""

    @classmethod
    def get_recovery_action(
        cls,
        error_context: ErrorContext,
        available_fallbacks: Dict[str, Callable] = None
    ) -> RecoveryAction:
        """Determine the appropriate recovery action for an error."""

        available_fallbacks = available_fallbacks or {}

        # Critical errors - abort immediately
        if error_context.severity == ErrorSeverity.CRITICAL:
            return RecoveryAction(action_type="abort")

        # Category-specific recovery strategies
        if error_context.error_category == ErrorCategory.RATE_LIMIT:
            # Exponential backoff for rate limits
            delay = min(60, 2 ** error_context.attempt_number) + random.uniform(0, 5)
            return RecoveryAction(
                action_type="retry",
                delay_seconds=delay
            )

        elif error_context.error_category == ErrorCategory.TIMEOUT:
            # Retry with exponential backoff
            delay = min(30, 2 ** (error_context.attempt_number - 1))
            return RecoveryAction(
                action_type="retry",
                delay_seconds=delay
            )

        elif error_context.error_category == ErrorCategory.NETWORK_ERROR:
            # Retry with linear backoff
            delay = min(20, error_context.attempt_number * 3)
            return RecoveryAction(
                action_type="retry",
                delay_seconds=delay
            )

        elif error_context.error_category == ErrorCategory.MODEL_ERROR:
            # Try fallback model if available
            if 'fallback_model' in available_fallbacks:
                return RecoveryAction(
                    action_type="fallback",
                    fallback_function=available_fallbacks['fallback_model']
                )
            else:
                return RecoveryAction(action_type="skip")

        elif error_context.error_category == ErrorCategory.QUOTA_EXCEEDED:
            # Try alternative provider if available
            if 'alternative_provider' in available_fallbacks:
                return RecoveryAction(
                    action_type="fallback",
                    fallback_function=available_fallbacks['alternative_provider']
                )
            else:
                return RecoveryAction(action_type="abort")

        elif error_context.error_category == ErrorCategory.PROCESSING_ERROR:
            # Retry with simplified parameters
            if error_context.attempt_number < 3:
                return RecoveryAction(
                    action_type="retry",
                    delay_seconds=1
                )
            elif 'simplified_processing' in available_fallbacks:
                return RecoveryAction(
                    action_type="fallback",
                    fallback_function=available_fallbacks['simplified_processing']
                )
            else:
                return RecoveryAction(action_type="skip")

        elif error_context.error_category == ErrorCategory.VALIDATION_ERROR:
            # Try data cleaning if available
            if 'clean_data' in available_fallbacks:
                return RecoveryAction(
                    action_type="fallback",
                    fallback_function=available_fallbacks['clean_data']
                )
            else:
                return RecoveryAction(action_type="skip")

        # Default: retry with exponential backoff
        if error_context.attempt_number < error_context.max_attempts:
            delay = min(10, 2 ** (error_context.attempt_number - 1))
            return RecoveryAction(
                action_type="retry",
                delay_seconds=delay
            )
        else:
            return RecoveryAction(action_type="abort")


class ErrorRecoveryHandler:
    """Handles error recovery with comprehensive logging and monitoring."""

    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
        self.error_history: List[ErrorContext] = []
        self.recovery_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'failed_recoveries': 0,
            'category_counts': {},
            'severity_counts': {}
        }

    async def handle_with_recovery(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: Dict[str, Any] = None,
        fallback_functions: Dict[str, Callable] = None,
        context: Dict[str, Any] = None
    ) -> Any:
        """Execute a function with comprehensive error handling and recovery."""

        kwargs = kwargs or {}
        fallback_functions = fallback_functions or {}
        context = context or {}

        last_error = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                result = await func(*args, **kwargs)

                # Success - log recovery if this wasn't the first attempt
                if attempt > 1:
                    logger.info(f"Successfully recovered from error on attempt {attempt}")
                    self.recovery_stats['recovered_errors'] += 1

                return result

            except Exception as error:
                last_error = error
                self.recovery_stats['total_errors'] += 1

                # Analyze the error
                error_category = ErrorAnalyzer.categorize_error(error)
                severity = ErrorAnalyzer.assess_severity(error_category, attempt)

                # Update statistics
                self.recovery_stats['category_counts'][error_category] = \
                    self.recovery_stats['category_counts'].get(error_category, 0) + 1
                self.recovery_stats['severity_counts'][severity] = \
                    self.recovery_stats['severity_counts'].get(severity, 0) + 1

                # Create error context
                error_context = ErrorContext(
                    function_name=func.__name__,
                    input_data={"args": args, "kwargs": kwargs},
                    attempt_number=attempt,
                    max_attempts=self.max_attempts,
                    error_category=error_category,
                    severity=severity,
                    timestamp=time.time(),
                    additional_info=context
                )

                # Store error history
                self.error_history.append(error_context)

                # Keep only recent errors
                if len(self.error_history) > 100:
                    self.error_history = self.error_history[-100:]

                # Log the error
                logger.warning(
                    f"Error in {func.__name__} (attempt {attempt}/{self.max_attempts}): "
                    f"{error_category.value} - {str(error)}"
                )

                # Determine recovery action
                recovery_action = RecoveryStrategy.get_recovery_action(
                    error_context, fallback_functions
                )

                # Execute recovery action
                if recovery_action.action_type == "abort":
                    logger.error(f"Aborting due to {severity.value} error: {str(error)}")
                    self.recovery_stats['failed_recoveries'] += 1
                    raise error

                elif recovery_action.action_type == "skip":
                    logger.warning(f"Skipping due to unrecoverable error: {str(error)}")
                    return None

                elif recovery_action.action_type == "fallback":
                    try:
                        logger.info(f"Attempting fallback recovery for {error_category.value}")
                        result = await recovery_action.fallback_function(
                            *recovery_action.fallback_args,
                            **recovery_action.fallback_kwargs
                        )
                        self.recovery_stats['recovered_errors'] += 1
                        return result
                    except Exception as fallback_error:
                        logger.warning(f"Fallback failed: {str(fallback_error)}")
                        # Continue to retry logic

                elif recovery_action.action_type == "retry":
                    if attempt < self.max_attempts:
                        if recovery_action.delay_seconds > 0:
                            logger.info(
                                f"Retrying in {recovery_action.delay_seconds:.1f} seconds..."
                            )
                            await asyncio.sleep(recovery_action.delay_seconds)
                        continue
                    else:
                        logger.error(f"Max attempts reached for {func.__name__}")
                        self.recovery_stats['failed_recoveries'] += 1
                        raise error

        # Should not reach here, but handle gracefully
        self.recovery_stats['failed_recoveries'] += 1
        raise last_error

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error handling statistics."""
        total_errors = self.recovery_stats['total_errors']

        return {
            'total_errors': total_errors,
            'recovered_errors': self.recovery_stats['recovered_errors'],
            'failed_recoveries': self.recovery_stats['failed_recoveries'],
            'recovery_rate': (
                self.recovery_stats['recovered_errors'] / max(1, total_errors)
            ),
            'category_breakdown': self.recovery_stats['category_counts'],
            'severity_breakdown': self.recovery_stats['severity_counts'],
            'recent_errors': [
                {
                    'function': ctx.function_name,
                    'category': ctx.error_category.value,
                    'severity': ctx.severity.value,
                    'attempt': ctx.attempt_number,
                    'timestamp': ctx.timestamp
                }
                for ctx in self.error_history[-10:]
            ]
        }


def resilient_async(
    max_attempts: int = 3,
    fallback_functions: Dict[str, Callable] = None,
    context: Dict[str, Any] = None
):
    """Decorator for adding resilient error handling to async functions."""

    def decorator(func: Callable):
        error_handler = ErrorRecoveryHandler(max_attempts)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await error_handler.handle_with_recovery(
                func, args, kwargs, fallback_functions, context
            )

        # Attach error handler for statistics
        wrapper.error_handler = error_handler
        return wrapper

    return decorator


@asynccontextmanager
async def error_monitoring_context(name: str):
    """Context manager for monitoring errors in a code block."""
    start_time = time.time()
    errors_caught = []

    try:
        logger.info(f"Starting monitored operation: {name}")
        yield errors_caught

    except Exception as e:
        errors_caught.append({
            'error': str(e),
            'type': type(e).__name__,
            'category': ErrorAnalyzer.categorize_error(e),
            'timestamp': time.time()
        })
        logger.error(f"Error in monitored operation {name}: {str(e)}")
        raise

    finally:
        duration = time.time() - start_time
        logger.info(
            f"Monitored operation {name} completed in {duration:.2f}s "
            f"with {len(errors_caught)} errors"
        )


class HealthCheck:
    """System health monitoring and validation."""

    def __init__(self):
        self.health_metrics = {
            'api_status': {},
            'error_rates': {},
            'performance_metrics': {},
            'last_check': 0
        }

    async def check_api_health(self, provider: str, test_func: Callable) -> Dict[str, Any]:
        """Check the health of an API provider."""
        start_time = time.time()

        try:
            await asyncio.wait_for(test_func(), timeout=30.0)

            health_status = {
                'status': 'healthy',
                'response_time': time.time() - start_time,
                'error': None,
                'timestamp': time.time()
            }

        except Exception as e:
            health_status = {
                'status': 'unhealthy',
                'response_time': time.time() - start_time,
                'error': str(e),
                'error_category': ErrorAnalyzer.categorize_error(e).value,
                'timestamp': time.time()
            }

        self.health_metrics['api_status'][provider] = health_status
        return health_status

    async def comprehensive_health_check(
        self,
        api_tests: Dict[str, Callable]
    ) -> Dict[str, Any]:
        """Perform comprehensive health check across all systems."""

        health_results = {}

        # Check all APIs
        for provider, test_func in api_tests.items():
            health_results[provider] = await self.check_api_health(provider, test_func)

        # Overall system health
        healthy_apis = sum(1 for result in health_results.values() if result['status'] == 'healthy')
        total_apis = len(health_results)

        overall_health = {
            'overall_status': 'healthy' if healthy_apis == total_apis else 'degraded' if healthy_apis > 0 else 'unhealthy',
            'healthy_apis': healthy_apis,
            'total_apis': total_apis,
            'health_percentage': (healthy_apis / max(1, total_apis)) * 100,
            'individual_status': health_results,
            'timestamp': time.time()
        }

        self.health_metrics['last_check'] = time.time()

        return overall_health

    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary."""
        return {
            'last_check_time': self.health_metrics['last_check'],
            'api_status': self.health_metrics['api_status'],
            'system_uptime': time.time() - self.health_metrics.get('start_time', time.time())
        }


# Global error handler instance
global_error_handler = ErrorRecoveryHandler()

# Convenience functions
async def safe_execute(func: Callable, *args, **kwargs) -> Any:
    """Safely execute a function with global error handling."""
    return await global_error_handler.handle_with_recovery(func, args, kwargs)


def get_global_error_stats() -> Dict[str, Any]:
    """Get global error handling statistics."""
    return global_error_handler.get_error_statistics()