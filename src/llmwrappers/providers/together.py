from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion import ChatCompletion

from ..oai_wrapper import OAIWrapper


class TogetherWrapper(OAIWrapper):
    """
    An asynchronous implementation of ChatWrapper for interacting with Together AI models.

    This class provides an asynchronous interface for querying Together AI language models
    using their OpenAI-compatible API.

    Attributes:
        model (str): The name of the Together AI model to use.
        client (AsyncOpenAI): An instance of the asynchronous OpenAI client configured for Together AI.

    Inherits from:
        OAIWrapper: Base class for OpenAI-compatible API wrappers
    """

    def __init__(self, model: str, *args, **kwargs) -> None:
        """
        Initialize the TogetherWrapper.

        Args:
            model (str): The name of the Together AI model to use.
            *args: Variable length argument list to pass to the parent constructor.
            **kwargs: Arbitrary keyword arguments to pass to the parent constructor.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.client = AsyncOpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=kwargs.get("api_key")
        )

    async def create(
        self, **kwargs
    ) -> AsyncStream[ChatCompletion] | ChatCompletion:
        """
        Create a chat completion using the Together AI API.

        Args:
            **kwargs: Keyword arguments to pass to the Together AI API.
                     The model name will be automatically added.

        Returns:
            ChatCompletion: The completion response from the Together AI API.
        """
        kwargs["model"] = self.model
        return await self.client.chat.completions.create(**kwargs)
