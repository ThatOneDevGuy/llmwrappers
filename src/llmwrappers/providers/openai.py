from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion import ChatCompletion

from ..oai_wrapper import OAIWrapper


class OpenAIWrapper(OAIWrapper):
    """
    An asynchronous implementation of ChatEngine for interacting with OpenAI models.

    This class provides an asynchronous interface for querying OpenAI language models.

    Attributes:
        model (str): The name of the OpenAI model to use.
        client (AsyncOpenAI): An instance of the asynchronous OpenAI client for API interactions.

    """

    def __init__(self, model: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.client = AsyncOpenAI()

    async def create(
        self, **kwargs
    ) -> AsyncStream[ChatCompletion] | ChatCompletion:
        kwargs["model"] = self.model
        return await self.client.chat.completions.create(**kwargs)
