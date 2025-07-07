from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion import ChatCompletion

from ..oai_wrapper import OAIWrapper


class GroqWrapper(OAIWrapper):
    """
    An asynchronous implementation of ChatWrapper for interacting with Groq models.

    This class provides an asynchronous interface for querying Groq language models.

    Attributes:
        model (str): The name of the Groq model to use.
        client (AsyncOpenAI): An instance of the asynchronous OpenAI client configured for Groq.

    Inherits from:
        OAIWrapper: Base class for OpenAI-compatible API wrappers
    """

    def __init__(self, model: str, *args, **kwargs):
        """
        Initialize the GroqWrapper.

        Args:
            model (str): The name of the Groq model to use.
            *args: Variable length argument list to pass to the parent constructor.
            **kwargs: Arbitrary keyword arguments to pass to the parent constructor.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=kwargs.get("api_key")
        )

    async def create(
        self, **kwargs
    ) -> AsyncStream[ChatCompletion] | ChatCompletion:
        """
        Create a chat completion using the Groq API.

        Args:
            **kwargs: Keyword arguments to pass to the Groq API.
                     The model name will be automatically added.

        Returns:
            ChatCompletion: The completion response from the Groq API.
        """
        kwargs["model"] = self.model

        if "stream_options" in kwargs:
            del kwargs["stream_options"]

        return await self.client.chat.completions.create(**kwargs)
