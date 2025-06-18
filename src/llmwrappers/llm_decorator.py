"""Module providing a decorator pattern implementation for LLM wrappers.

This module implements the decorator pattern to allow for dynamic modification and enhancement
of LLM wrapper behavior. It enables hooking into query lifecycle events and modifying
both prompt and API arguments before they are passed to the underlying LLM.
"""

from typing import AsyncGenerator, Type, TypeVar, Union

from pydantic import BaseModel

from .base_wrapper import LLMWrapper

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class LLMDecorator(LLMWrapper):
    underlying_llm: LLMWrapper = None

    def __init__(self, underlying_llm: LLMWrapper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.underlying_llm = underlying_llm

    async def query_response(self, **kwargs) -> str:
        """Execute a query and return the response as a string.

        This method allows hooking into the query lifecycle through the hook_query method.
        Arguments with uppercase keys are treated as prompt arguments, while others are API arguments.

        Args:
            **kwargs: Keyword arguments for both prompt and API configuration

        Returns:
            str: The response from the LLM
        """
        lifecycle = self._hook_all_args(**kwargs)
        args = await lifecycle.asend(None)

        result = await self.underlying_llm.query_response(**args)

        try:
            await lifecycle.asend(result)
        except RuntimeError as e:
            if isinstance(e.__cause__, StopAsyncIteration):
                pass
            else:
                raise e

        return result

    async def query_object(self, response_model: Type[T], **kwargs) -> T:
        """Execute a query and parse the response into a Pydantic model.

        Args:
            response_model: The Pydantic model class to parse the response into
            **kwargs: Keyword arguments for both prompt and API configuration

        Returns:
            T: An instance of the specified response_model containing the parsed response
        """
        lifecycle = self._hook_all_args(**kwargs)
        args = await lifecycle.asend(None)

        result = await self.underlying_llm.query_object(response_model, **args)

        try:
            await lifecycle.asend(result)
        except RuntimeError as e:
            if isinstance(e.__cause__, StopAsyncIteration):
                pass
            else:
                raise e

        return result

    async def query_block(self, block_type: str, **kwargs) -> str:
        """Execute a query and extract a specific block type from the response.

        Args:
            block_type: The type of block to extract from the response
            **kwargs: Keyword arguments for both prompt and API configuration

        Returns:
            str: The extracted block content from the response
        """
        lifecycle = self._hook_all_args(**kwargs)
        args = await lifecycle.asend(None)

        result = await self.underlying_llm.query_block(block_type, **args)

        try:
            await lifecycle.asend(result)
        except RuntimeError as e:
            if isinstance(e.__cause__, StopAsyncIteration):
                pass
            else:
                raise e

        return result

    async def query_stream(
        self, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Execute a streaming query that yields response chunks.

        This method streams the response from the LLM, collecting the chunks to pass
        the complete response to the lifecycle hooks once streaming is complete.

        Args:
            **kwargs: Keyword arguments for both prompt and API configuration

        Yields:
            str: Response chunks as they become available
        """
        lifecycle = self._hook_all_args(**kwargs)
        args = await lifecycle.asend(None)

        result = []
        async for chunk in self.underlying_llm.query_stream(**args):
            result.append(chunk)
            yield chunk

        try:
            await lifecycle.asend("".join(result))
        except RuntimeError as e:
            if isinstance(e.__cause__, StopAsyncIteration):
                pass
            else:
                raise e

    async def hook_query(self, prompt_args: dict[str, str], api_args: dict[str, str]) -> AsyncGenerator[tuple[dict[str, str], dict[str, str]], str]:
        """Hook into the query lifecycle to modify arguments and process responses.

        This method can be overridden by subclasses to implement custom behavior.
        The default implementation passes through the arguments unchanged.

        Args:
            prompt_args: Dictionary of prompt-related arguments (uppercase keys)
            api_args: Dictionary of API-related arguments (lowercase keys)

        Yields:
            Initial yield: Modified combined arguments dictionary
            Final yield: None (after receiving the response)

        Returns:
            AsyncGenerator yielding modified arguments and receiving the response
        """
        response = yield {**prompt_args, **api_args}

    async def _hook_all_args(self, **kwargs) -> AsyncGenerator[dict[str, str], str]:
        """Internal method to process all arguments through the query lifecycle hooks.

        This method separates prompt and API arguments based on key case, then passes
        them through the hook_query lifecycle for potential modification.

        Args:
            **kwargs: Combined prompt and API arguments

        Returns:
            AsyncGenerator yielding modified arguments and receiving the response
        """
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        lifecycle = self.hook_query(prompt_args, api_args)
        new_kwargs = await lifecycle.asend(None)
        response = yield new_kwargs
        await lifecycle.asend(response)

