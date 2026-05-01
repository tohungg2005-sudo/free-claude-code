"""Application services for the Claude-compatible API."""

from __future__ import annotations

import asyncio
import contextlib
import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from config.settings import Settings
from core.anthropic import get_token_count, get_user_facing_error_message
from core.anthropic.sse import ANTHROPIC_SSE_RESPONSE_HEADERS
from providers.base import BaseProvider
from providers.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
)

from .gas_client import get_or_keys, log_request
from .model_router import ModelRouter
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.responses import TokenCountResponse
from .optimization_handlers import try_optimizations
from .web_tools.egress import WebFetchEgressPolicy
from .web_tools.request import (
    is_web_server_tool_request,
    openai_chat_upstream_server_tool_error,
)
from .web_tools.streaming import stream_web_server_tool_response

TokenCounter = Callable[[list[Any], str | list[Any] | None, list[Any] | None], int]

ProviderGetter = Callable[[str], BaseProvider]

# Providers that use ``/chat/completions`` + Anthropic-to-OpenAI conversion (not native Messages).
_OPENAI_CHAT_UPSTREAM_IDS = frozenset({"nvidia_nim"})


class KeyRotator:
    """Xoay vòng API keys khi bị rate limit, thread-safe."""

    def __init__(self, keys: list[str]):
        self._keys = keys
        self._index = 0

    def current(self) -> str | None:
        if not self._keys:
            return None
        return self._keys[self._index % len(self._keys)]

    def next(self) -> str | None:
        """Chuyển sang key tiếp theo. Trả về None nếu đã hết vòng."""
        if not self._keys:
            return None
        self._index += 1
        if self._index >= len(self._keys):
            self._index = 0
            return None  # Đã thử hết tất cả keys
        return self._keys[self._index]

    def all_keys(self) -> list[str]:
        return list(self._keys)


# Backward compat alias
ORKeyRotator = KeyRotator


def anthropic_sse_streaming_response(
    body: AsyncIterator[str],
) -> StreamingResponse:
    """Return a :class:`StreamingResponse` for Anthropic-style SSE streams."""
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers=ANTHROPIC_SSE_RESPONSE_HEADERS,
    )


def _http_status_for_unexpected_service_exception(_exc: BaseException) -> int:
    """HTTP status for uncaught non-provider failures (stable client contract)."""
    return 500


def _log_unexpected_service_exception(
    settings: Settings,
    exc: BaseException,
    *,
    context: str,
    request_id: str | None = None,
) -> None:
    """Log service-layer failures without echoing exception text unless opted in."""
    if settings.log_api_error_tracebacks:
        if request_id is not None:
            logger.error("{} request_id={}: {}", context, request_id, exc)
        else:
            logger.error("{}: {}", context, exc)
        logger.error(traceback.format_exc())
        return
    if request_id is not None:
        logger.error(
            "{} request_id={} exc_type={}",
            context,
            request_id,
            type(exc).__name__,
        )
    else:
        logger.error("{} exc_type={}", context, type(exc).__name__)


def _require_non_empty_messages(messages: list[Any]) -> None:
    if not messages:
        raise InvalidRequestError("messages cannot be empty")


class ClaudeProxyService:
    """Coordinate request optimization, model routing, token count, and providers."""

    def __init__(
        self,
        settings: Settings,
        provider_getter: ProviderGetter,
        model_router: ModelRouter | None = None,
        token_counter: TokenCounter = get_token_count,
    ):
        self._settings = settings
        self._provider_getter = provider_getter
        self._model_router = model_router or ModelRouter(settings)
        self._token_counter = token_counter
        # Key rotator mặc định từ .env (OPENROUTER_API_KEYS plural)
        self._default_or_keys = settings.get_openrouter_keys()
        self._default_rotator = KeyRotator(self._default_or_keys)
        # Cache OR key rotator riêng theo từng license key
        self._license_rotators: dict[str, KeyRotator] = {}
        # NIM key rotator từ .env (NVIDIA_NIM_API_KEYS plural)
        self._nim_rotator = KeyRotator(settings.get_nim_keys())

    def _get_provider_with_key(self, api_key: str) -> BaseProvider:
        """Tạo OpenRouter provider với key cụ thể."""
        from config.provider_catalog import OPENROUTER_DEFAULT_BASE
        from providers.base import ProviderConfig
        from providers.open_router import OpenRouterProvider

        config = ProviderConfig(
            api_key=api_key,
            base_url=OPENROUTER_DEFAULT_BASE,
            rate_limit=self._settings.provider_rate_limit,
            rate_window=self._settings.provider_rate_window,
            max_concurrency=self._settings.provider_max_concurrency,
            http_read_timeout=self._settings.http_read_timeout,
            http_write_timeout=self._settings.http_write_timeout,
            http_connect_timeout=self._settings.http_connect_timeout,
            enable_thinking=self._settings.enable_model_thinking,
            proxy=self._settings.open_router_proxy,
            log_raw_sse_events=self._settings.log_raw_sse_events,
            log_api_error_tracebacks=self._settings.log_api_error_tracebacks,
        )
        return OpenRouterProvider(config)

    def _get_nim_provider_with_key(self, api_key: str) -> BaseProvider:
        """Tạo NVIDIA NIM provider với key cụ thể."""
        from providers.base import ProviderConfig
        from providers.nvidia_nim import NVIDIA_NIM_DEFAULT_BASE, NvidiaNimProvider

        config = ProviderConfig(
            api_key=api_key,
            base_url=NVIDIA_NIM_DEFAULT_BASE,
            rate_limit=self._settings.provider_rate_limit,
            rate_window=self._settings.provider_rate_window,
            max_concurrency=self._settings.provider_max_concurrency,
            http_read_timeout=self._settings.http_read_timeout,
            http_write_timeout=self._settings.http_write_timeout,
            http_connect_timeout=self._settings.http_connect_timeout,
            enable_thinking=self._settings.enable_model_thinking,
            proxy=self._settings.nvidia_nim_proxy,
            log_raw_sse_events=self._settings.log_raw_sse_events,
            log_api_error_tracebacks=self._settings.log_api_error_tracebacks,
        )
        return NvidiaNimProvider(config, nim_settings=self._settings.nim)

    def _try_nim_key_rotation(
        self,
        routed_request: Any,
        thinking_enabled: bool,
        input_tokens: int,
    ) -> StreamingResponse | None:
        """Thử lần lượt tất cả NIM keys còn lại trong rotator."""
        remaining_keys = (
            self._nim_rotator.all_keys()[self._nim_rotator._index + 1 :]
            if self._nim_rotator._index + 1 < len(self._nim_rotator._keys)
            else []
        )
        for key in remaining_keys:
            self._nim_rotator.next()
            logger.warning("NIM_KEY_ROTATE: thử key tiếp theo ({}...)", key[:12])
            try:
                alt_provider = self._get_nim_provider_with_key(key)
                alt_provider.preflight_stream(
                    routed_request,
                    thinking_enabled=thinking_enabled,
                )
                request_id = f"req_{uuid.uuid4().hex[:12]}"
                return anthropic_sse_streaming_response(
                    alt_provider.stream_response(
                        routed_request,
                        input_tokens=input_tokens,
                        request_id=request_id,
                        thinking_enabled=thinking_enabled,
                    )
                )
            except ProviderError:
                continue
        return None

    def _get_rotator(self, license_key: str) -> KeyRotator:
        """Trả về ORKeyRotator của license key (đọc từ cache RAM, không gọi GAS sync)."""
        if license_key and license_key in self._license_rotators:
            return self._license_rotators[license_key]
        return self._default_rotator

    async def _load_rotator_async(self, license_key: str) -> None:
        """Load OR keys từ GAS cho license key và lưu vào cache (chạy background)."""
        if not license_key or license_key in self._license_rotators:
            return
        or_keys = await get_or_keys(license_key)
        if or_keys:
            self._license_rotators[license_key] = ORKeyRotator(or_keys)
            logger.info(
                "OR_ROTATOR_LOADED: license={} keys={}", license_key, len(or_keys)
            )
        else:
            self._license_rotators[license_key] = ORKeyRotator(self._default_or_keys)

    # Status codes that trigger key rotation / model fallback
    _FALLBACK_STATUS_CODES = frozenset({401, 402, 429})

    def _is_fallback_error(self, exc: ProviderError) -> bool:
        return (
            isinstance(exc, (AuthenticationError, RateLimitError))
            or exc.status_code in self._FALLBACK_STATUS_CODES
        )

    def _try_or_key_rotation(
        self,
        rotator: ORKeyRotator,
        routed_request: Any,
        thinking_enabled: bool,
        input_tokens: int,
    ) -> StreamingResponse | None:
        """
        Thử lần lượt tất cả OR keys còn lại trong rotator.
        Trả về StreamingResponse nếu một key thành công, None nếu hết key.

        Bug 1 fix: gọi preflight_stream() để raise lỗi NGAY (không lazy),
        tránh việc lỗi xảy ra sau khi đã trả response về client.
        """
        remaining_keys = (
            rotator.all_keys()[rotator._index + 1 :]
            if rotator._index + 1 < len(rotator._keys)
            else []
        )
        for key in remaining_keys:
            rotator.next()
            logger.warning("OR_KEY_ROTATE: thử key tiếp theo ({}...)", key[:12])
            try:
                alt_provider = self._get_provider_with_key(key)
                # preflight_stream raise ProviderError ngay nếu key lỗi
                alt_provider.preflight_stream(
                    routed_request,
                    thinking_enabled=thinking_enabled,
                )
                request_id = f"req_{uuid.uuid4().hex[:12]}"
                return anthropic_sse_streaming_response(
                    alt_provider.stream_response(
                        routed_request,
                        input_tokens=input_tokens,
                        request_id=request_id,
                        thinking_enabled=thinking_enabled,
                    )
                )
            except ProviderError:
                continue  # key này cũng lỗi → thử tiếp
        return None

    def create_message(
        self, request_data: MessagesRequest, *, license_key: str = ""
    ) -> object:
        """Create a message response or streaming response."""
        try:
            _require_non_empty_messages(request_data.messages)

            routed = self._model_router.resolve_messages_request(request_data)
            provider_id = routed.resolved.provider_id

            if provider_id in _OPENAI_CHAT_UPSTREAM_IDS:
                tool_err = openai_chat_upstream_server_tool_error(
                    routed.request,
                    web_tools_enabled=self._settings.enable_web_server_tools,
                )
                if tool_err is not None:
                    raise InvalidRequestError(tool_err)

            if self._settings.enable_web_server_tools and is_web_server_tool_request(
                routed.request
            ):
                input_tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                logger.info("Optimization: Handling Anthropic web server tool")
                egress = WebFetchEgressPolicy(
                    allow_private_network_targets=self._settings.web_fetch_allow_private_networks,
                    allowed_schemes=self._settings.web_fetch_allowed_scheme_set(),
                )
                return anthropic_sse_streaming_response(
                    stream_web_server_tool_response(
                        routed.request,
                        input_tokens=input_tokens,
                        web_fetch_egress=egress,
                        verbose_client_errors=self._settings.log_api_error_tracebacks,
                    ),
                )

            optimized = try_optimizations(routed.request, self._settings)
            if optimized is not None:
                return optimized
            logger.debug("No optimization matched, routing to provider")

            provider = self._provider_getter(provider_id)
            provider.preflight_stream(
                routed.request,
                thinking_enabled=routed.resolved.thinking_enabled,
            )

            request_id = f"req_{uuid.uuid4().hex[:12]}"

            # Lấy OR rotator đúng cho license key (Bug 2 fix: không ghi đè rotator chung)
            rotator = self._get_rotator(license_key)
            if license_key and license_key not in self._license_rotators:
                # Load async ở background, lần này dùng default
                with contextlib.suppress(RuntimeError):
                    asyncio.ensure_future(self._load_rotator_async(license_key))

            current_or_key = rotator.current()
            key_hint = f"({current_or_key[:12]}...)" if current_or_key else "(no key)"
            logger.info(
                "✅ REQUEST: id={} provider={} model={} key={}",
                request_id,
                provider_id,
                routed.request.model,
                key_hint,
            )
            if self._settings.log_raw_api_payloads:
                logger.debug(
                    "FULL_PAYLOAD [{}]: {}", request_id, routed.request.model_dump()
                )

            input_tokens = self._token_counter(
                routed.request.messages, routed.request.system, routed.request.tools
            )

            # Log request vào Google Sheet (fire-and-forget)
            log_request(license_key, routed.request.model)

            return anthropic_sse_streaming_response(
                provider.stream_response(
                    routed.request,
                    input_tokens=input_tokens,
                    request_id=request_id,
                    thinking_enabled=routed.resolved.thinking_enabled,
                ),
            )

        except ProviderError as e:
            if self._is_fallback_error(e):
                input_tokens = self._token_counter(
                    routed.request.messages,
                    routed.request.system,
                    routed.request.tools,
                )
                # Bước 1a: Thử xoay NIM key nếu đang dùng nvidia_nim
                if routed.resolved.provider_id == "nvidia_nim":
                    rotated = self._try_nim_key_rotation(
                        routed.request,
                        routed.resolved.thinking_enabled,
                        input_tokens,
                    )
                    if rotated is not None:
                        return rotated

                # Bước 1b: Thử xoay OR key nếu đang dùng open_router
                if routed.resolved.provider_id == "open_router":
                    rotator = self._get_rotator(license_key)
                    rotated = self._try_or_key_rotation(
                        rotator,
                        routed.request,
                        routed.resolved.thinking_enabled,
                        input_tokens,
                    )
                    if rotated is not None:
                        return rotated

                # Bước 2: Fallback sang model tiếp theo trong chain
                # Bug 3 fix: so sánh provider_model_ref (đã route) với fallback chain
                fallback_chain = self._settings.get_fallback_chain()
                current_ref = (
                    routed.resolved.provider_model_ref
                )  # vd: "nvidia_nim/meta/llama-3.3-70b-instruct"
                try:
                    idx = fallback_chain.index(current_ref)
                    next_models = fallback_chain[idx + 1 :]
                except ValueError:
                    next_models = fallback_chain

                if next_models:
                    next_model = next_models[0]
                    logger.warning(
                        "MODEL_FALLBACK: status={} model={} -> {}",
                        e.status_code,
                        current_ref,
                        next_model,
                    )
                    fallback_request = request_data.model_copy(deep=True)
                    fallback_request.model = next_model
                    return self.create_message(
                        fallback_request, license_key=license_key
                    )
            raise
        except Exception as e:
            _log_unexpected_service_exception(
                self._settings, e, context="CREATE_MESSAGE_ERROR"
            )
            raise HTTPException(
                status_code=_http_status_for_unexpected_service_exception(e),
                detail=get_user_facing_error_message(e),
            ) from e

    def count_tokens(self, request_data: TokenCountRequest) -> TokenCountResponse:
        """Count tokens for a request after applying configured model routing."""
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        with logger.contextualize(request_id=request_id):
            try:
                _require_non_empty_messages(request_data.messages)
                routed = self._model_router.resolve_token_count_request(request_data)
                tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                logger.info(
                    "COUNT_TOKENS: request_id={} model={} messages={} input_tokens={}",
                    request_id,
                    routed.request.model,
                    len(routed.request.messages),
                    tokens,
                )
                return TokenCountResponse(input_tokens=tokens)
            except ProviderError:
                raise
            except Exception as e:
                _log_unexpected_service_exception(
                    self._settings,
                    e,
                    context="COUNT_TOKENS_ERROR",
                    request_id=request_id,
                )
                raise HTTPException(
                    status_code=_http_status_for_unexpected_service_exception(e),
                    detail=get_user_facing_error_message(e),
                ) from e
