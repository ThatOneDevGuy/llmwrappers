import sys
import traceback
import inspect
from typing import get_type_hints

from mcp import Tool
from pydantic import create_model


def get_exception_details():
    """
    Gets the most recent exception's details including stack trace.
    Returns a tuple of (exception_type, exception_value, formatted_traceback)
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if exc_type is None:  # No exception is being handled
        return None

    # Get formatted traceback
    formatted_trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))

    return {
        'ExceptionType': exc_type.__name__,
        'ExceptionValue': str(exc_value),
        'Traceback': formatted_trace
    }


def get_stack_source_code():
    """
    Gets the source code for each function in the current exception's stack trace.
    Returns a list of tuples: (function_name, filename, line_number, source_code)
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if exc_type is None:
        return None

    stack_info = []
    tb = exc_traceback
    while tb:
        frame = tb.tb_frame
        function = frame.f_code.co_name
        filename = frame.f_code.co_filename
        line_number = tb.tb_lineno

        try:
            # Get the source code for the function
            source_lines, start_line = inspect.getsourcelines(frame.f_code)
            source_code = ''.join(source_lines)
        except (IOError, TypeError):
            source_code = "Source code not available"

        stack_info.append({
            'Function': function,
            'Filename': filename,
            'LineNumber': line_number,
            'SourceCode': source_code
        })
        tb = tb.tb_next

    return stack_info


def create_model_from_function(func):
    """Creates a Pydantic model from a function's signature."""
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    fields = {}
    for name, param in sig.parameters.items():
        annotation = type_hints.get(name, type(None))
        default = ... if param.default is param.empty else param.default
        fields[name] = (annotation, default)

    model_name = f"{func.__name__.title()}Arguments"
    return create_model(model_name, **fields)
