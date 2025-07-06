from fireworks.client.api import ChatCompletionResponse, CompletionStreamResponse
from typing import AsyncGenerator
import warnings
from openai import AsyncOpenAI

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fireworks.client import AsyncFireworks

import os
from ..oai_wrapper import OAIWrapper


class FireworksWrapper(OAIWrapper):
    """
    An asynchronous implementation of ChatEngine for interacting with Fireworks models.

    This class provides an asynchronous interface for querying Fireworks language models.

    Attributes:
        model (str): The name of the Fireworks model to use.
        client (AsyncFireworks): An instance of the asynchronous Fireworks client for API interactions.

    """

    def __init__(self, model: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        # self.client = AsyncFireworks(api_key=os.getenv("FIREWORKS_API_KEY"))
        self.client = AsyncOpenAI(
            api_key=os.getenv("FIREWORKS_API_KEY"),
            base_url="https://api.fireworks.ai/inference/v1"
        )

    async def create(
        self, **kwargs
    ) -> AsyncGenerator[CompletionStreamResponse, None] | ChatCompletionResponse:
        kwargs["model"] = self.model
        return await self.client.chat.completions.create(**kwargs)
