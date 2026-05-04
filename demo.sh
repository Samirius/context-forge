#!/usr/bin/env bash
# Quick start demo for context-forge
# This script creates a playbook, seeds it, and starts the server

set -e

DB="${CTXF_DB_PATH:-$HOME/.ctxf/demo.db}"
HOST="${CTXF_HOST:-0.0.0.0}"
PORT="${CTXF_PORT:-8000}"

echo "=== context-forge Quick Start Demo ==="
echo ""

# Init
echo "1. Creating playbook..."
ctxf init --name demo --db "$DB"

# Seed
echo ""
echo "2. Seeding with example data..."
ctxf seed --domain general --db "$DB"

# List
echo ""
echo "3. Current playbook contents:"
ctxf list --db "$DB"

# Stats
echo ""
echo "4. Playbook stats:"
ctxf stats --db "$DB"

# Test retrieval
echo ""
echo "5. Testing retrieval for 'API error handling':"
ctxf retrieve "API error handling" --db "$DB"

# Export markdown
echo ""
echo "6. Exporting playbook as markdown:"
ctxf export --format markdown --db "$DB"

echo ""
echo "=== Starting API server on http://${HOST}:${PORT} ==="
echo "Try: curl http://localhost:${PORT}/health"
echo "Try: curl http://localhost:${PORT}/v1/playbook"
echo ""
echo "Press Ctrl+C to stop"

CTXF_DB_PATH="$DB" ctxf serve --host "$HOST" --port "$PORT"
