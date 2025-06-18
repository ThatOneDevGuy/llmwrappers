from ..chat_wrapper import ChatWrapper

import fastapi_poe as fp


class PoeWrapper(ChatWrapper):
    def __init__(self, model: str, request: fp.QueryRequest, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.request = request

    async def query(self, **kwargs):
        request = self.request.model_copy()

        request.query = [
            fp.ProtocolMessage(
                role=x['role'],
                content=x['content'],
            )
            for x in kwargs.get('messages', [])
        ]

        request.temperature = kwargs.get('temperature', None)

        if not kwargs.get('stream', False):
            async for chunk in fp.stream_request(request, self.model, request.access_key):
                yield chunk.text
        else:
            response = []
            async for chunk in fp.stream_request(request, self.model, request.access_key):
                response.append(chunk.text)
            yield ''.join(response)
            
