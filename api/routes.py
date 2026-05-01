"""FastAPI route handlers."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from loguru import logger

from config.settings import Settings
from core.anthropic import get_token_count

from . import dependencies
from .dependencies import get_settings, require_api_key
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.responses import ModelResponse, ModelsListResponse
from .services import ClaudeProxyService


def _extract_license_key(request: Request) -> str:
    """Lấy license key từ header x-api-key (bỏ qua phần sau dấu ':')."""
    header = request.headers.get("x-api-key") or request.headers.get(
        "authorization", ""
    )
    if header.lower().startswith("bearer "):
        header = header.split(" ", 1)[1]
    # Token format: LICENSE_KEY:model_name → lấy phần đầu
    return header.split(":", 1)[0].strip()


router = APIRouter()


SUPPORTED_CLAUDE_MODELS = [
    ModelResponse(
        id="claude-opus-4-20250514",
        display_name="Claude Opus 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-haiku-4-20250514",
        display_name="Claude Haiku 4",
        created_at="2025-05-14T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-opus-20240229",
        display_name="Claude 3 Opus",
        created_at="2024-02-29T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        created_at="2024-10-22T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-haiku-20240307",
        display_name="Claude 3 Haiku",
        created_at="2024-03-07T00:00:00Z",
    ),
    ModelResponse(
        id="claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku",
        created_at="2024-10-22T00:00:00Z",
    ),
]


def get_proxy_service(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> ClaudeProxyService:
    """Build the request service for route handlers."""
    return ClaudeProxyService(
        settings,
        provider_getter=lambda provider_type: dependencies.resolve_provider(
            provider_type, app=request.app, settings=settings
        ),
        token_counter=get_token_count,
    )


def _probe_response(allow: str) -> Response:
    """Return an empty success response for compatibility probes."""
    return Response(status_code=204, headers={"Allow": allow})


# =============================================================================
# Routes
# =============================================================================
@router.post("/v1/messages")
async def create_message(
    request: Request,
    request_data: MessagesRequest,
    service: ClaudeProxyService = Depends(get_proxy_service),
    _auth=Depends(require_api_key),
):
    """Create a message (always streaming)."""
    license_key = _extract_license_key(request)
    return service.create_message(request_data, license_key=license_key)


@router.api_route("/v1/messages", methods=["HEAD", "OPTIONS"])
async def probe_messages(_auth=Depends(require_api_key)):
    """Respond to Claude compatibility probes for the messages endpoint."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.post("/v1/messages/count_tokens")
async def count_tokens(
    request_data: TokenCountRequest,
    service: ClaudeProxyService = Depends(get_proxy_service),
    _auth=Depends(require_api_key),
):
    """Count tokens for a request."""
    return service.count_tokens(request_data)


@router.api_route("/v1/messages/count_tokens", methods=["HEAD", "OPTIONS"])
async def probe_count_tokens(_auth=Depends(require_api_key)):
    """Respond to Claude compatibility probes for the token count endpoint."""
    return _probe_response("POST, HEAD, OPTIONS")


@router.get("/")
async def root(
    settings: Settings = Depends(get_settings), _auth=Depends(require_api_key)
):
    """Root endpoint."""
    return {
        "status": "ok",
        "provider": settings.provider_type,
        "model": settings.model,
    }


@router.api_route("/", methods=["HEAD", "OPTIONS"])
async def probe_root(_auth=Depends(require_api_key)):
    """Respond to compatibility probes for the root endpoint."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.api_route("/health", methods=["HEAD", "OPTIONS"])
async def probe_health():
    """Respond to compatibility probes for the health endpoint."""
    return _probe_response("GET, HEAD, OPTIONS")


@router.get("/v1/models", response_model=ModelsListResponse)
async def list_models(_auth=Depends(require_api_key)):
    """List the Claude model ids this proxy advertises for compatibility."""
    return ModelsListResponse(
        data=SUPPORTED_CLAUDE_MODELS,
        first_id=SUPPORTED_CLAUDE_MODELS[0].id if SUPPORTED_CLAUDE_MODELS else None,
        has_more=False,
        last_id=SUPPORTED_CLAUDE_MODELS[-1].id if SUPPORTED_CLAUDE_MODELS else None,
    )


@router.post("/next-key")
async def next_or_key(
    _auth=Depends(require_api_key),
):
    """Chuyển sang OR key tiếp theo trong OPENROUTER_API_KEYS.

    Đọc trực tiếp từ .env (không qua settings cache) để luôn thấy
    key hiện tại đúng → xoay sang key kế tiếp → ghi lại vào .env.
    """
    import os
    from pathlib import Path

    from dotenv import dotenv_values

    env_path = Path.home() / ".config" / "free-claude-code" / ".env"
    if not env_path.exists():
        env_path = Path(os.environ.get("FCC_ENV_FILE", ".env"))

    try:
        # Đọc trực tiếp từ file, không qua cache
        env_vals = dotenv_values(env_path)
        keys_raw = env_vals.get("OPENROUTER_API_KEYS", "")
        keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
        current = env_vals.get("OPENROUTER_API_KEY", "").strip()

        if len(keys) <= 1:
            return {"status": "only_one", "message": "Chi co 1 key, khong the xoay"}

        try:
            idx = keys.index(current)
            next_key = keys[(idx + 1) % len(keys)]
        except ValueError:
            next_key = keys[0]

        lines = env_path.read_text().splitlines()
        for i, line in enumerate(lines):
            if (
                line.strip().startswith("OPENROUTER_API_KEY=")
                and "OPENROUTER_API_KEYS" not in line
            ):
                lines[i] = f'OPENROUTER_API_KEY="{next_key}"'
                break
        env_path.write_text("\n".join(lines))
        logger.info(
            "NEXT_KEY: {}...{} -> {}...{}",
            current[:8],
            current[-4:],
            next_key[:8],
            next_key[-4:],
        )
        return {
            "status": "ok",
            "message": f"Da chuyen sang key {next_key[:8]}...{next_key[-4:]}",
        }
    except Exception as exc:
        logger.error("NEXT_KEY_ERROR: {}", exc)
        return {"status": "error", "message": str(exc)}


@router.post("/stop")
async def stop_cli(request: Request, _auth=Depends(require_api_key)):
    """Stop all CLI sessions and pending tasks."""
    handler = getattr(request.app.state, "message_handler", None)
    if not handler:
        # Fallback if messaging not initialized
        cli_manager = getattr(request.app.state, "cli_manager", None)
        if cli_manager:
            await cli_manager.stop_all()
            logger.info("STOP_CLI: source=cli_manager cancelled_count=N/A")
            return {"status": "stopped", "source": "cli_manager"}
        raise HTTPException(status_code=503, detail="Messaging system not initialized")

    count = await handler.stop_all_tasks()
    logger.info("STOP_CLI: source=handler cancelled_count={}", count)
    return {"status": "stopped", "cancelled_count": count}
