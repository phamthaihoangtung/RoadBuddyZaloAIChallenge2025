FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# uv install
COPY --from=ghcr.io/astral-sh/uv:0.5.26 /uv /bin/uv

COPY pyproject.toml /code/pyproject.toml

ENV UV_HTTP_TIMEOUT=300
WORKDIR /code
RUN uv venv --python=3.10
RUN uv sync 
RUN uv pip install unsloth jupyterlab av

COPY models /code/models/
COPY configs /code/configs
COPY src /code/src
COPY predict.sh /code/predict.sh