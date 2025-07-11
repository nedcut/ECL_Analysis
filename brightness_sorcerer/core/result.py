"""Result type for better error handling and functional programming patterns."""

from typing import TypeVar, Generic, Union, Callable, Optional, Any
from dataclasses import dataclass
import logging

from .exceptions import BrightnessSorcererError

T = TypeVar('T')
E = TypeVar('E', bound=Exception)
U = TypeVar('U')


@dataclass
class Success(Generic[T]):
    """Represents a successful result."""
    value: T
    
    def is_success(self) -> bool:
        return True
    
    def is_error(self) -> bool:
        return False
    
    def unwrap(self) -> T:
        """Get the success value."""
        return self.value
    
    def unwrap_or(self, default: T) -> T:
        """Get the success value (ignores default)."""
        return self.value
    
    def map(self, func: Callable[[T], U]) -> 'Result[U, E]':
        """Apply function to success value."""
        try:
            return Success(func(self.value))
        except Exception as e:
            return Error(e)
    
    def and_then(self, func: Callable[[T], 'Result[U, E]']) -> 'Result[U, E]':
        """Chain operations that return Results."""
        try:
            return func(self.value)
        except Exception as e:
            return Error(e)
    
    def or_else(self, func: Callable[[E], 'Result[T, E]']) -> 'Result[T, E]':
        """Return self (no error to handle)."""
        return self


@dataclass
class Error(Generic[E]):
    """Represents an error result."""
    error: E
    
    def is_success(self) -> bool:
        return False
    
    def is_error(self) -> bool:
        return True
    
    def unwrap(self) -> Any:
        """Raise the error (should not be called on Error)."""
        if isinstance(self.error, Exception):
            raise self.error
        else:
            raise RuntimeError(f"Result contains error: {self.error}")
    
    def unwrap_or(self, default: T) -> T:
        """Return the default value."""
        return default
    
    def map(self, func: Callable[[Any], U]) -> 'Result[U, E]':
        """Return self (no value to map)."""
        return self
    
    def and_then(self, func: Callable[[Any], 'Result[U, E]']) -> 'Result[U, E]':
        """Return self (no value to chain)."""
        return self
    
    def or_else(self, func: Callable[[E], 'Result[T, E]']) -> 'Result[T, E]':
        """Handle the error with a recovery function."""
        try:
            return func(self.error)
        except Exception as e:
            return Error(e)


# Type alias for convenience
Result = Union[Success[T], Error[E]]


def success(value: T) -> Success[T]:
    """Create a successful result."""
    return Success(value)


def error(err: E) -> Error[E]:
    """Create an error result."""
    return Error(err)


def safe_call(func: Callable[..., T], *args, **kwargs) -> Result[T, Exception]:
    """
    Safely call a function and return a Result.
    
    Args:
        func: Function to call
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Success with return value or Error with exception
    """
    try:
        result = func(*args, **kwargs)
        return success(result)
    except Exception as e:
        logging.debug(f"safe_call caught exception: {e}")
        return error(e)


def safe_call_with_log(func: Callable[..., T], operation_name: str, 
                      *args, **kwargs) -> Result[T, Exception]:
    """
    Safely call a function with automatic logging.
    
    Args:
        func: Function to call
        operation_name: Name of operation for logging
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Success with return value or Error with exception
    """
    try:
        logging.debug(f"Starting operation: {operation_name}")
        result = func(*args, **kwargs)
        logging.debug(f"Operation completed successfully: {operation_name}")
        return success(result)
    except Exception as e:
        logging.error(f"Operation failed: {operation_name} - {e}")
        return error(e)


def validate_and_convert(value: Any, validator: Callable[[Any], bool], 
                        converter: Callable[[Any], T], 
                        error_message: str) -> Result[T, BrightnessSorcererError]:
    """
    Validate and convert a value.
    
    Args:
        value: Value to validate and convert
        validator: Function that returns True if value is valid
        converter: Function to convert value to target type
        error_message: Error message if validation fails
        
    Returns:
        Success with converted value or Error with validation error
    """
    try:
        if not validator(value):
            return error(BrightnessSorcererError(error_message, f"Value: {value}"))
        
        converted = converter(value)
        return success(converted)
    except Exception as e:
        return error(BrightnessSorcererError(error_message, f"Conversion failed: {e}", e))


class ResultBuilder:
    """Builder pattern for chaining Result operations."""
    
    def __init__(self, initial_result: Result[T, E]):
        self.result = initial_result
    
    def map(self, func: Callable[[T], U], operation_name: str = "") -> 'ResultBuilder':
        """Apply a mapping function."""
        if self.result.is_success():
            try:
                new_value = func(self.result.unwrap())
                self.result = success(new_value)
                if operation_name:
                    logging.debug(f"ResultBuilder map succeeded: {operation_name}")
            except Exception as e:
                self.result = error(e)
                if operation_name:
                    logging.error(f"ResultBuilder map failed: {operation_name} - {e}")
        return self
    
    def and_then(self, func: Callable[[T], Result[U, E]], operation_name: str = "") -> 'ResultBuilder':
        """Chain a function that returns a Result."""
        if self.result.is_success():
            try:
                self.result = func(self.result.unwrap())
                if operation_name and self.result.is_success():
                    logging.debug(f"ResultBuilder and_then succeeded: {operation_name}")
                elif operation_name:
                    logging.error(f"ResultBuilder and_then failed: {operation_name}")
            except Exception as e:
                self.result = error(e)
                if operation_name:
                    logging.error(f"ResultBuilder and_then threw exception: {operation_name} - {e}")
        return self
    
    def validate(self, predicate: Callable[[T], bool], error_msg: str) -> 'ResultBuilder':
        """Validate the current value."""
        if self.result.is_success():
            if not predicate(self.result.unwrap()):
                self.result = error(BrightnessSorcererError(error_msg))
                logging.error(f"ResultBuilder validation failed: {error_msg}")
        return self
    
    def log_success(self, message: str) -> 'ResultBuilder':
        """Log a message if the result is successful."""
        if self.result.is_success():
            logging.info(message)
        return self
    
    def log_error(self, message: str) -> 'ResultBuilder':
        """Log a message if the result is an error."""
        if self.result.is_error():
            logging.error(f"{message}: {self.result.error}")
        return self
    
    def unwrap_or_else(self, default_func: Callable[[], T]) -> T:
        """Get the value or call default function."""
        if self.result.is_success():
            return self.result.unwrap()
        else:
            return default_func()
    
    def build(self) -> Result[T, E]:
        """Get the final result."""
        return self.result


def result_builder(initial_value: T) -> ResultBuilder:
    """Create a new ResultBuilder with a success value."""
    return ResultBuilder(success(initial_value))


def error_builder(initial_error: E) -> ResultBuilder:
    """Create a new ResultBuilder with an error."""
    return ResultBuilder(error(initial_error))


# Utility functions for common patterns
def collect_results(results: list[Result[T, E]]) -> Result[list[T], list[E]]:
    """
    Collect a list of Results into a single Result.
    
    Returns Success with list of values if all are successful,
    or Error with list of errors if any failed.
    """
    successes = []
    errors = []
    
    for result in results:
        if result.is_success():
            successes.append(result.unwrap())
        else:
            errors.append(result.error)
    
    if errors:
        return error(errors)
    else:
        return success(successes)


def first_success(results: list[Result[T, E]]) -> Result[T, list[E]]:
    """
    Return the first successful result, or all errors if none succeed.
    """
    errors = []
    
    for result in results:
        if result.is_success():
            return result
        else:
            errors.append(result.error)
    
    return error(errors)