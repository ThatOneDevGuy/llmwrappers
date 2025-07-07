import os

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion import ChatCompletion

from ..oai_wrapper import OAIWrapper


class GrokWrapper(OAIWrapper):
    """
    An asynchronous implementation of ChatWrapper for interacting with Grok models.

    This class provides an asynchronous interface for querying Grok language models.

    Attributes:
        model (str): The name of the Grok model to use.
        client (AsyncOpenAI): An instance of the asynchronous OpenAI client configured for Grok.

    Inherits from:
        OAIWrapper: Base class for OpenAI-compatible API wrappers
    """

    def __init__(self, model: str, *args, **kwargs):
        """
        Initialize the GrokWrapper.

        Args:
            model (str): The name of the Grok model to use.
            *args: Variable length argument list to pass to the parent constructor.
            **kwargs: Arbitrary keyword arguments to pass to the parent constructor.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.client = AsyncOpenAI(
            base_url="https://api.x.ai/v1",
            api_key=kwargs.get("api_key") or os.environ.get("GROK_API_KEY"),
        )

    async def create(
        self, **kwargs
    ) -> AsyncStream[ChatCompletion] | ChatCompletion:
        """
        Create a chat completion using the Grok API.

        Args:
            **kwargs: Keyword arguments to pass to the Grok API.
                     The model name will be automatically added.

        Returns:
            ChatCompletion: The completion response from the Grok API.
        """
        kwargs["model"] = self.model
        return await self.client.chat.completions.create(**kwargs)
