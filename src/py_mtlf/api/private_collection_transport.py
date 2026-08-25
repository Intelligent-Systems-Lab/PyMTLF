from py_mtlf.api.problems import problem_response


class PrivateCollectionBodyMiddleware:
    """Bound private collection JSON bodies before FastAPI model parsing."""

    _LIMITS = {
        "/internal/v1/training-data-collections": 64 * 1024,
        "/callbacks/upf-event-exposure": 4 * 1024 * 1024,
    }

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        limit = self._LIMITS.get(scope["path"])
        if limit is None:
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        media_type = headers.get(b"content-type", b"").decode("latin-1").split(";", 1)[0]
        if media_type.strip().lower() != "application/json":
            await self._problem(
                scope,
                receive,
                send,
                415,
                "UNSUPPORTED_MEDIA_TYPE",
                "Content-Type must be application/json",
            )
            return
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > limit:
                    await self._problem(
                        scope,
                        receive,
                        send,
                        413,
                        "REQUEST_TOO_LARGE",
                        "request body exceeds the configured transport limit",
                    )
                    return
            except ValueError:
                pass
        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                await self.app(scope, self._single_message(message), send)
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > limit:
                await self._problem(
                    scope,
                    receive,
                    send,
                    413,
                    "REQUEST_TOO_LARGE",
                    "request body exceeds the configured transport limit",
                )
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        sent = False

        async def replay_receive():
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay_receive, send)

    @staticmethod
    def _single_message(message):
        async def receive():
            return message

        return receive

    @staticmethod
    async def _problem(scope, receive, send, status_code, cause, detail) -> None:
        response = problem_response(
            status_code,
            "Request rejected",
            detail,
            cause=cause,
        )
        await response(scope, receive, send)
