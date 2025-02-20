from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Generator, Optional, Type, TypeVar, Union

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class LLMFacade(ABC):
    """
    Base class for asynchronous Language Learning Model (LLM) engines.

    Provides a standardized interface for:
    - Making raw queries to LLM APIs
    - Getting responses as structured objects
    - Getting responses in specific block formats
    - Rate limiting and usage tracking

    Subclasses must implement the abstract query methods for specific LLM APIs.
    """


    @abstractmethod
    async def query_response(self, **kwargs: Any) -> tuple[str, int]:
        """
        Send a query to the LLM and get the complete response.

        Args:
            **kwargs: API-specific arguments (e.g. max_tokens, temperature)

        Returns:
            tuple[str, int]: Response text and tokens consumed
        """
        ...

    @abstractmethod
    async def query_stream(self, **kwargs: Any) -> AsyncGenerator[str, None]:
        """
        Send a query to the LLM and stream the response chunks.

        Args:
            **kwargs: API-specific arguments

        Yields:
            str: Response text chunks as they arrive
        """
        ...

    @abstractmethod
    async def query_object(self, response_model: Type[T], **kwargs: Any) -> T:
        """
        Query the LLM and parse the response into a structured object.

        Args:
            response_model: Pydantic model class to parse response into
            **kwargs: Prompt args (UPPERCASE) and API args (lowercase)

        Returns:
            T: Response parsed into response_model instance
        """
        ...

    @abstractmethod
    async def query_block(self, block_type: str, **kwargs: Any) -> str:
        """
        Query the LLM for a specific block type response.

        Args:
            block_type: Type of block to request (e.g. "python", "json")
            **kwargs: Prompt args (UPPERCASE) and API args (lowercase)

        Returns:
            str: Response formatted as requested block type
        """
        ...
