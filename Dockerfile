FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml .
COPY ctxf/ ctxf/

# Install dependencies
RUN uv pip install --system -e .

# Create data directory
RUN mkdir -p /data

ENV CTXF_DB_PATH=/data/playbook.db
ENV CTXF_HOST=0.0.0.0
ENV CTXF_PORT=8000

EXPOSE 8000

ENTRYPOINT ["ctxf"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
