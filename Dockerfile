# blog-agent 生产镜像：uv + venv，产出轻量运行时
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# 先只拷依赖清单，最大化利用层缓存
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 再拷源码
COPY app ./app
COPY scripts ./scripts
RUN uv sync --frozen --no-dev


FROM python:3.12-slim

WORKDIR /app
# 固定 UID/GID：宿主机如需 bind mount 可对应 chown 1000:1000
RUN groupadd -g 1000 app && useradd -u 1000 -g 1000 -m app

COPY --from=builder /app /app
RUN mkdir -p /app/data && chown -R app:app /app
USER app

EXPOSE 8000
# --proxy-headers + forwarded-allow-ips：信任同机 nginx，限流才能拿到真实客户端 IP
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips=127.0.0.1"]
