from typing import AsyncGenerator

from anthropic import AsyncAnthropic

from ..base_wrapper import LLMMetrics, tracks_metrics
from ..chat_wrapper import ChatWrapper


class AnthropicWrapper(ChatWrapper):
    """
    An asynchronous implementation of LLMWrapper for interacting with Anthropic models.

    This class provides an asynchronous interface for querying Anthropic language models.

    Attributes:
        model (str): The name of the Anthropic model to use.
        client (AsyncAnthropic): An instance of the asynchronous Anthropic client for API interactions.

    Inherits from:
        ChatWrapper
    """

    def __init__(self, model: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.client = AsyncAnthropic()

    @tracks_metrics
    async def query(
        self, *, metrics: LLMMetrics, **kwargs
    ) -> AsyncGenerator[str, None]:
        system_msg = []
        messages = kwargs["messages"]
        for i in reversed(range(len(messages))):
            msg = messages[i]
            if msg.get("role") == "system":
                system_msg.append(msg["content"])
                del messages[i]

        if "max_tokens" not in kwargs or kwargs["max_tokens"] is None:
            kwargs["max_tokens"] = 2**12

        kwargs["model"] = self.model

        if not kwargs.get("stream", False):
            kwargs.pop("stream", None)
            response = await self.client.messages.create(
                system="\n\n".join(system_msg),
                **kwargs,
            )

            total_tokens = response.usage.input_tokens + response.usage.output_tokens
            metrics.tokens_consumed = total_tokens
            yield response.content[0].text
        else:
            kwargs.pop("stream", None)

            if "stream_options" in kwargs:
                del kwargs["stream_options"]

            input_tokens = 0
            output_tokens = 0

            async with self.client.messages.stream(
                system="\n\n".join(system_msg), **kwargs
            ) as stream:
                async for chunk in stream:
                    if hasattr(chunk, "usage"):
                        if hasattr(chunk.usage, "input_tokens"):
                            input_tokens = chunk.usage.input_tokens
                        if hasattr(chunk.usage, "output_tokens"):
                            output_tokens = chunk.usage.output_tokens
                        metrics.tokens_consumed = input_tokens + output_tokens

                    if chunk.type not in ("content_block_start", "content_block_delta"):
                        continue

                    if hasattr(chunk, "content_block"):
                        if (
                            hasattr(chunk.content_block, "text")
                            and chunk.content_block.text
                        ):
                            yield chunk.content_block.text

                    if hasattr(chunk, "delta"):
                        if hasattr(chunk.delta, "text") and chunk.delta.text:
                            yield chunk.delta.text
