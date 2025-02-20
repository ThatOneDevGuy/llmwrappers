"""Example of using LLM Facades for contextual exception handling and debugging.

This example demonstrates creating an LLM Facade focused on debugging exceptions,
that automatically includes exception context when invoked. The facade has access
to:

- Full exception details including type, message and stack trace
- Source code of functions in the stack trace

This pattern is useful for:
- Interactive debugging with LLM assistance
- Automated error analysis and troubleshooting
- Generating human-readable explanations of errors
- Suggesting potential fixes based on full context

The facade pattern makes it easy to use this contextual information with any LLM
while maintaining a simple interface - the context is automatically injected
without the caller needing to handle it explicitly.
"""


from typing import AsyncGenerator
from .llm_facade import LLMDecorator
from .cerebras_facade import CerebrasFacade

import traceback
import inspect
import sys

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

class ExceptionQA(LLMDecorator):
    async def hook_query(self, prompt_args: dict[str, str], api_args: dict[str, str]) -> AsyncGenerator[tuple[dict[str, str], dict[str, str]], str]:
        exception_details = get_exception_details()
        code = get_stack_source_code()

        response = yield {
            "USER_ARGS": prompt_args,
            "EXCEPTION_DETAILS": exception_details,
            "STACK_SOURCE_CODE": code,
            "TASK": "Use the EXCEPTION_DETAILS and STACK_SOURCE_CODE to respond to the user's task in USER_ARGS.",
            **api_args
        }

def do_hard_math():
    return 1 / 0
 

async def main():
    gpt4omini = CerebrasFacade(model="llama-3.3-70b")
    exception_qa = ExceptionQA(underlying_llm=gpt4omini)

    try:
        do_hard_math()
    except Exception as e:
        print(await exception_qa.query_block(
            "md", # Output in markdown
            QUERY="Why am I getting this error?"
        ))
    

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
