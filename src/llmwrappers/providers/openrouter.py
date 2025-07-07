import os

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion import ChatCompletion

from ..oai_wrapper import OAIWrapper


class OpenRouterWrapper(OAIWrapper):
    """
    An asynchronous implementation of ChatWrapper for interacting with OpenRouter models.

    This class provides an asynchronous interface for querying OpenRouter language models.

    Attributes:
        model (str): The name of the OpenRouter model to use.
        client (AsyncOpenAI): An instance of the asynchronous OpenAI client configured for OpenRouter.

    Inherits from:
        OAIWrapper: Base class for OpenAI-compatible API wrappers
    """

    def __init__(self, model: str, *args, url: str = "https://openrouter.ai/api/v1", **kwargs):
        """
        Initialize the OpenRouterWrapper.

        Args:
            model (str): The name of the OpenRouter model to use.
            *args: Variable length argument list to pass to the parent constructor.
            url (str): The OpenRouter API URL. Defaults to "https://openrouter.ai/api/v1".
            **kwargs: Arbitrary keyword arguments to pass to the parent constructor.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.client = AsyncOpenAI(
            base_url=url,
            api_key=kwargs.get("api_key") or os.environ.get("OPENROUTER_API_KEY"),
        )

    async def create(
        self, **kwargs
    ) -> AsyncStream[ChatCompletion] | ChatCompletion:
        """
        Create a chat completion using the OpenRouter API.

        Args:
            **kwargs: Keyword arguments to pass to the OpenRouter API.
                     The model name will be automatically added.

        Returns:
            ChatCompletion: The completion response from the OpenRouter API.
        """
        kwargs["model"] = self.model
        return await self.client.chat.completions.create(**kwargs)
