from abc import ABC, abstractmethod
import asyncio
import json
from typing import AsyncGenerator, Type, TypeVar, Any


from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel, TypeAdapter, validate_call


from .base_wrapper import LLMMetrics, tracks_metrics

from .chat_wrapper import ChatWrapper
from .wrapper_utils import compile_user_prompt, compile_to_string, generate_schema

from .tools import get_exception_details, get_stack_source_code, create_model_from_function, Tool
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

T = TypeVar("T", bound=BaseModel)


class OAIWrapper(ChatWrapper, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    async def create(self, **kwargs) -> ChatCompletion: ...

    @tracks_metrics
    async def query(
        self, *, metrics: LLMMetrics, **kwargs
    ) -> AsyncGenerator[str, None]:
        api_args = {k: v for k, v in kwargs.items() if k.upper() != k}
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}

        if "tool_choice" in api_args:
            raise NotImplementedError("Tool choice not implemented for OAIEngine")

        if prompt_args:
            api_args.setdefault("messages", []).extend(
                {"role": "user", "content": await compile_user_prompt(**prompt_args)}
            )

        if not api_args.get("stream", False):
            response: ChatCompletion = await self.create(**api_args)

            tokens_consumed = response.usage.total_tokens

            # Check if the conversation was too long for the context window
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                raise Exception("The conversation was too long for the context window.")

            # Check if the model's output included copyright material (or similar)
            if finish_reason == "content_filter":
                raise Exception("Content was filtered due to policy violations.")

            # Else the model is responding directly to the user
            if finish_reason in ("stop", "eos"):
                metrics.tokens_consumed += tokens_consumed
                yield response.choices[0].message.content

            # Catch any other case, this is unexpected
            else:
                raise Exception("Unexpected API finish_reason:", finish_reason)
        else:
            api_args["stream_options"] = {"include_usage": True}

            async for chunk in await self.create(**api_args):
                if metrics is not None:
                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        # With include_usage, the final chunk object should include the total tokens consumed
                        # So we can override our default assumption of 1 token on the final chunk
                        metrics.tokens_consumed = chunk.usage.total_tokens
                    else:
                        # By default, assume 1 token per chunk
                        metrics.tokens_consumed += 1

                if chunk.choices:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

    @tracks_metrics
    async def query(self, *, metrics: LLMMetrics, **kwargs) -> AsyncGenerator[str, None]:
        api_args = {k: v for k, v in kwargs.items() if k.upper() != k}
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        tools = {}

        call_limit = api_args.pop('call_limit', None)
        if call_limit is not None:
            assert call_limit >= 0

        if 'tools' in api_args:
            schemas = []
            for tool in api_args.pop('tools'):
                if isinstance(tool, Tool):
                    function = {
                        'name': tool.name,
                        'description': tool.description,
                        'parameters': {
                            **tool.inputSchema,
                            'additionalProperties': False
                        },
                    }
                else:
                    function = {
                        'name': tool.__name__,
                        'description': tool.__doc__,
                        'parameters': {
                            **create_model_from_function(tool).model_json_schema(),
                            'additionalProperties': False
                        },
                    }

                tool_schema = {
                    'type': 'function',
                    'function': function,
                }
                schemas.append(tool_schema)
                tools[tool.__name__] = validate_call(tool)

            api_args['tools'] = schemas
            if 'tool_choice' not in api_args:
                api_args['tool_choice'] = 'auto'

        if prompt_args:
            api_args.setdefault("messages", []).extend(
                {"role": "user", "content": await compile_user_prompt(**prompt_args)}
            )
        
        async def handle_tool_call(tool_calls):
            api_args["messages"].append(ChatCompletionMessage(
                role='assistant',
                content=None,
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id=x.id,
                        function=Function(
                            arguments=x.function.arguments,
                            name=x.function.name,
                        ),
                        type="function"
                    )
                    for x in tool_calls
                ]
            ))

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                try:
                    if asyncio.iscoroutinefunction(tools[tool_name]):
                        tool_result = await tools[tool_name](**tool_args)
                    else:
                        tool_result = tools[tool_name](**tool_args)
                    

                    tool_result_str = await compile_to_string(tool_result)
                except Exception as e:
                    tool_result_str = await compile_to_string({
                        "RESULT": "Error, did not complete successfully",
                        "EXCEPTION_DETAILS": get_exception_details(),
                        "STACK": get_stack_source_code()
                    })


                api_args["messages"].append({
                    'role': 'tool',
                    "name": tool_name,
                    "tool_call_id": tool_call.id,
                    "content": tool_result_str,
                })

            return await self.create(**api_args)
        
        async def collect_response(stream):
            result = []
            finish_reason = None

            tool_calls = []
            current_tool_id = None
            current_tool_name = None
            current_tool_args = []

            async for chunk in stream:
                if metrics is not None:
                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        # With include_usage, the final chunk object should include the total tokens consumed
                        # So we can override our default assumption of 1 token on the final chunk
                        metrics.tokens_consumed += chunk.usage.total_tokens
                    else:
                        # By default, assume 1 token per chunk
                        metrics.tokens_consumed += 1

                if chunk.choices:
                    if chunk.choices[0].delta.content:
                        result.append(chunk.choices[0].delta.content)
                    
                    finish_reason = finish_reason or chunk.choices[0].finish_reason
                    
                    for tool_call in chunk.choices[0].delta.tool_calls or []:
                        if tool_call.id:
                            if tool_call.id != current_tool_id:
                                if current_tool_id:
                                    tool_calls.append(ChatCompletionMessageToolCall(
                                        id=current_tool_id,
                                        function=Function(
                                            arguments=''.join(current_tool_args),
                                            name=current_tool_name,
                                        ),
                                        type='function',
                                    ))
                            current_tool_id = tool_call.id
                            current_tool_name = tool_call.function.name
                            current_tool_args = [tool_call.function.arguments or '']
                        
                        current_tool_name = current_tool_name or tool_call.function.name
                        current_tool_args.append(tool_call.function.arguments or '')
            
            if current_tool_id:
                tool_calls.append(ChatCompletionMessageToolCall(
                    id=current_tool_id,
                    function=Function(
                        arguments=''.join(current_tool_args),
                        name=current_tool_name,
                    ),
                    type='function',
                ))
            
            if finish_reason == 'tool_calls':
                return tool_calls, finish_reason
            
            response = ''.join(result)
            return response, finish_reason


        if not api_args.get("stream", False):
            response: ChatCompletion = await self.create(**api_args)
            finish_reason = response.choices[0].finish_reason

            while finish_reason == 'tool_calls':                    
                if call_limit is not None:
                    if call_limit <= 0:
                        raise Exception("Tool call limit exceeded")
                    call_limit -= 1
                response = await handle_tool_call(response.choices[0].message.tool_calls)
                finish_reason = response.choices[0].finish_reason

            # Check if the conversation was too long for the context window
            if finish_reason == "length":
                raise Exception("The conversation was too long for the context window.")

            # Check if the model's output included copyright material (or similar)
            if finish_reason == "content_filter":
                raise Exception("Content was filtered due to policy violations.")

            # Else the model is responding directly to the user
            if finish_reason in ("stop", "eos"):
                if metrics is not None:
                    metrics.tokens_consumed += response.usage.total_tokens
                yield response.choices[0].message.content

            # Catch any other case, this is unexpected
            else:
                raise Exception("Unexpected API finish_reason:", finish_reason)
        else:
            api_args["stream_options"] = {"include_usage": True}

            if 'tools' not in api_args:
                async for chunk in await self.create(**api_args):
                    if metrics is not None:
                        if hasattr(chunk, "usage") and chunk.usage is not None:
                            # With include_usage, the final chunk object should include the total tokens consumed
                            # So we can override our default assumption of 1 token on the final chunk
                            metrics.tokens_consumed = chunk.usage.total_tokens
                        else:
                            # By default, assume 1 token per chunk
                            metrics.tokens_consumed += 1

                    if chunk.choices:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
            else:                
                stream = await self.create(**api_args)
                response, finish_reason = await collect_response(stream)
                
                while finish_reason == 'tool_calls':                    
                    if call_limit is not None:
                        if call_limit <= 0:
                            raise Exception("Tool call limit exceeded")
                        call_limit -= 1
                    stream = await handle_tool_call(response)
                    response, finish_reason = await collect_response(stream)

                # Check if the conversation was too long for the context window
                if finish_reason == "length":
                    raise Exception("The conversation was too long for the context window.")

                # Check if the model's output included copyright material (or similar)
                if finish_reason == "content_filter":
                    raise Exception("Content was filtered due to policy violations.")

                # Else the model is responding directly to the user
                if finish_reason in ("stop", "eos"):
                    yield response

                # Catch any other case, this is unexpected
                else:
                    raise Exception("Unexpected API finish_reason:", finish_reason)
                

    async def query_object(self, response_model: Type[T], **kwargs) -> T:
        prompt_args = {k: v for k, v in kwargs.items() if k == k.upper()}
        api_args = {k: v for k, v in kwargs.items() if k != k.upper()}

        await self._update_messages_with_prompt_args(
            api_args.setdefault("messages", []), prompt_args
        )
        api_args.setdefault("messages", []).extend(
            await _generate_obj_query_messages(response_model)
        )

        schema = generate_schema(response_model)

        wrapped_schema = schema
        if schema['type'] != 'object':
            wrapped_schema = {
                'properties': {
                    'data': {
                        'title': 'Data', **schema
                    }
                },
                'required': ['data'],
                'title': 'Wrapper',
                'type': 'object'
            }

        stream = self.query(
            **api_args,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        **wrapped_schema,
                        "strict": True
                    }
                },
            }
        )

        result = []
        async for chunk in stream:
            result.append(chunk)

        response = "".join(result)
        response = json.loads(response)

        if schema != wrapped_schema:
            response = response['data']

        try:
            if issubclass(response_model, BaseModel):
                return response_model(**response)
        except TypeError:
            pass

        return TypeAdapter(response_model).validate_python(response)


async def _generate_obj_query_messages(
    response_model: Type[BaseModel] | Type[Any],
) -> list[dict[str, Any]]:
    """
    Generate messages for an object query.

    This function creates a system message and a user message for querying
    an LLM to generate a response matching a specific model.

    Args:
        response_model (BaseModel): The expected response model.

    Returns:
        list: A list of message dictionaries for the LLM query.
    """
    user_prompt = (
        "Return the correct JSON response, not the "
        "JSON_SCHEMA. Use only fields specified by the JSON_SCHEMA and nothing else."
    )

    schema = None
    try:
        if issubclass(response_model, BaseModel):
            schema = response_model.model_json_schema()
    except TypeError:
        pass

    if schema is None:
        schema = TypeAdapter(response_model).json_schema()

    system_prompt = (
        "Your task is to understand the content and provide "
        "the parsed objects in json that matches the following json_schema:\n\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Make sure to return an instance of the JSON, not the schema itself."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
