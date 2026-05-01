"""Proxy standalone entry point - được nhúng vào claude_v2.exe"""
import os
import sys

# Đọc config từ env vars (binary sẽ set trước khi chạy)
if __name__ == "__main__":
    import uvicorn
    from api.app import create_app

    app = create_app()
    port = int(os.environ.get("PROXY_PORT", "8082"))
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        timeout_graceful_shutdown=5,
    )
