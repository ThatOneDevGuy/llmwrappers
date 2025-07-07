# LLM Wrappers
This project introduces the concept of LLM Wrappers. LLM Wrappers are a lightweight design pattern for building scalable, composable complexity on top of LLMs. They provide a way to specialize LLM usage and encapsulate functionality so that it can be used as if it were just another LLM.

```python
try:
    ...
except Exception as e:
    # With LLM Wrappers:
    debugger_llm = ExceptionQA(...)
    debugger_llm.invoke("Why did I get the last exception?")

    # Without LLM Wrappers:
    llm = ChatModel(...)
    llm.invoke(
        "Why did I get the following exception?\n\n"
        "# Exception Details:\n"
        f"{get_exception_details()}\n\n"
        "# Stack Trace:\n"
        f"{get_stack_source_code()}\n\n"
    )
```

# Overview
LLM usage, and especially prompt construction, is complex, tedious, and error-prone. Agentic functionallity typically requires chaining many LLM calls together, which exacerbates these issues. While it's possible to wrap LLM calls within functions that handle prompt construction, this approach does not integrate cleanly with chat use cases since each constructed function has its own interface that requires special handling. For the same reason, this approach does not integrate cleanly with existing LLM interfaces or with applications where the LLM API can be provided as a blackbox endpoint.

LLM Wrappers address these issues in two ways:
1. It creates a standard interface for passing *prompt arguments* to LLMs as key-value pairs. This allows for easy prompt construction and encapsulation.
2. It exposes a standard LLM interface for callers. This allows LLM Wrappers to be used as a drop-in replacement for any LLM, which makes LLM Wrappers highly composable and highly interoperable with LLM applications both within and outside of existing agent frameworks.


# Using LLM Wrappers
The following code demonstrates the usage interface for LLM Wrappers. Every LLM Wrapper supports these methods. Note that:
- Prompt arguments are passed as uppercase keyword arguments, and API arguments are passed as lowercase or mixed case keyword arguments. This convention is used to avoid confusing API arguments (passed to the underlying LLM API) with prompt arguments (passed by the caller to parameterize prompts). Prompt arguments can be any JSON-serializable object, including Pydantic models.
- The task is passed as a prompt argument. Typically if only one task is required, its description is passed as the last prompt argument under a keyword like `TASK` or `PROMPT`. There is no special handling of the last keyword argument in any of the LLM Wrapper base classes. It's up to the underlying LLM to interpret this as the actual task to perform.

```python
from llmwrappers.providers.openai import OpenAIWrapper


async def main():
    llm = OpenAIWrapper(model="gpt-4o-mini")

    # Streaming responses
    async for chunk in llm.query_stream(
        PROMPT="How does modern photolithography exposure work?"
    ):
        print(chunk)
    
    # Non-streaming responses
    response = await llm.query_response(
        PROMPT="How does modern photolithography exposure work?"
    )
    print(response)

    # Structured outputs
    steps = await llm.query_object(
        list[str], # type annotations and Pydantic models are supported
        TASK="List out the steps of a modern photolithography exposure process."
    )
    print(steps)

    # Markdown blocks
    markdown = await llm.query_block(
        "python", # markdown block types recognized by the LLM are supported
        TASK="Give me python pseudocode explaining how modern photolithography exposure works."
    )
    print(markdown)

```


# Creating LLM Wrappers
The `llmwrapper` module provides two interfaces for creating LLM Wrappers. These are described below in increasing order of flexibility


## `ChatWrapper` base class
The `ChatWrapper` base class lets you create an LLM Wrapper from an LLM chat model. This requires only implementing a `query` method that accepts a chat history involving system and user roles and a flag for streaming responses. Regardless of whether the caller wants streaming responses, the `query` method yields a stream of chunks. In the case of non-streaming responses, it yields a single chunk with the entire response. All popular LLM providers expose an interface that's compatible with this class.

```python
class MyWrapper(ChatWrapper):
    async def query(self, stream, **kwargs):
        if stream:
            async for chunk in ...:
                yield chunk
        else:
            response = ...
            yield response

my_llm = MyWrapper(
    underlying_llm=wrapper_from_chatmodel(
        ChatOpenAI(model="gpt-4o-mini")
    )
)

print(await my_llm.query_response(
    TASK="How does modern photolithography exposure work?"
))
```

## `LLMWrapper` base class
The `LLMWrapper` base class lets you create an LLM Wrapper from scratch. This requires implementing `query_response` (complete responses), `query_stream` (streaming responses), `query_object` (structured outputs), and `query_block` (markdown blocks). It's very unlikely that you will need to extend `LLMWrapper` directly, but it's provided for cases where you need to implement an LLM Wrapper that's not based on a chat-based LLM.

```python
class MyWrapper(LLMWrapper):

    def __init__(self, underlying_llm: LLMWrapper, **kwargs):
        super().__init__(**kwargs)
        self.underlying_llm = underlying_llm

    async def query_response(self, **kwargs) -> str:
        return await self.underlying_llm.query_response(**kwargs)

    async def query_stream(self, **kwargs) -> AsyncGenerator[str, None]:
        return await self.underlying_llm.query_stream(**kwargs)

    async def query_object(self, response_model: Type[T], **kwargs) -> T:
        return await self.underlying_llm.query_object(response_model, **kwargs)

    async def query_block(self, block_type: str, **kwargs) -> str:
        return await self.underlying_llm.query_block(block_type, **kwargs)

my_llm = MyWrapper(
    underlying_llm=wrapper_from_chatmodel(
        ChatOpenAI(model="gpt-4o-mini")
    )
)

print(await my_llm.query_response(
    TASK="How does modern photolithography exposure work?"
))
```

