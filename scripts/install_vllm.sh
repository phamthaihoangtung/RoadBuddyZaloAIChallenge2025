export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}
export UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-3}

uv sync
uv pip install vllm --torch-backend=auto