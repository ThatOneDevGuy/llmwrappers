from typing import AsyncGenerator, Type, TypeVar, Union

from pydantic import BaseModel

from .llm_facade import LLMFacade

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class LLMDecorator(LLMFacade):
    underlying_llm: LLMFacade = None

    def __init__(self, underlying_llm: LLMFacade, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.underlying_llm = underlying_llm

    async def query_response(self, **kwargs) -> str:
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
        response = yield {**prompt_args, **api_args}

    async def _hook_all_args(self, **kwargs) -> AsyncGenerator[dict[str, str], str]:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        lifecycle = self.hook_query(prompt_args, api_args)
        new_kwargs = await lifecycle.asend(None)
        response = yield new_kwargs
        await lifecycle.asend(response)
