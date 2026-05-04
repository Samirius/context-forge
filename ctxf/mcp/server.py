"""MCP (Model Context Protocol) stdio server for context-forge.

Implements tools: retrieve, reflect, curate, evolve.
Communicates via JSON-RPC over stdin/stdout.
"""

from __future__ import annotations

import asyncio
import json
import sys
import logging
from typing import Any

from ctxf.config import get_settings
from ctxf.store.sqlite_store import SqliteStore
from ctxf.retrieval.embedder import Embedder
from ctxf.retrieval.hybrid import HybridRetriever
from ctxf.engine.reflector import Reflector
from ctxf.engine.curator import Curator
from ctxf.engine.generator import Generator
from ctxf.llm.factory import create_llm

logger = logging.getLogger(__name__)


# --- JSON-RPC helpers ---

def _json_rpc_response(id: int | str | None, result: Any) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": id, "result": result}) + "\n"


def _json_rpc_error(id: int | str | None, code: int, message: str) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}) + "\n"


# --- Tool definitions ---

TOOLS = [
    {
        "name": "retrieve",
        "description": "Search the playbook for relevant context bullets matching a query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Number of results", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "reflect",
        "description": "Analyze a task execution outcome and produce structured insights.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The original task"},
                "response": {"type": "string", "description": "The AI response"},
                "outcome": {"type": "string", "description": "Human feedback on outcome"},
            },
            "required": ["task", "response"],
        },
    },
    {
        "name": "curate",
        "description": "Convert a reflection into playbook delta operations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "reflection": {"type": "object", "description": "Structured reflection from reflect tool"},
                "auto_apply": {"type": "boolean", "description": "Auto-apply deltas", "default": true},
            },
            "required": ["reflection"],
        },
    },
    {
        "name": "evolve",
        "description": "Run the full ACE loop: retrieve → generate → reflect → curate → apply.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Task to execute"},
                "outcome": {"type": "string", "description": "Optional outcome feedback"},
                "auto_apply": {"type": "boolean", "description": "Auto-apply deltas", "default": true},
            },
            "required": ["task"],
        },
    },
]


async def _handle_tool_call(name: str, arguments: dict) -> Any:
    """Execute a tool and return the result."""
    settings = get_settings()
    store = SqliteStore(settings.db_path)

    try:
        pb = store.get_playbook()
        if not pb:
            return {"error": "No playbook found. Run 'ctxf init' first."}

        if name == "retrieve":
            embedder = Embedder(
                base_url=settings.effective_embed_base_url,
                api_key=settings.effective_embed_api_key,
                model=settings.embed_model,
                dim=settings.embed_dim,
            ) if settings.embeddings_enabled else None

            retriever = HybridRetriever(
                embedder=embedder,
                top_k=arguments.get("top_k", settings.retrieval_top_k),
                alpha=settings.hybrid_alpha,
            )
            results = await retriever.retrieve(arguments["query"], pb.active_bullets())
            return {
                "results": [
                    {"id": b.id, "section": b.section, "content": b.content, "score": round(s, 4)}
                    for b, s in results
                ]
            }

        elif name == "reflect":
            refl_llm = create_llm(settings, "reflector")
            reflector = Reflector(refl_llm)
            reflection = await reflector.reflect(
                task=arguments["task"],
                response=arguments["response"],
                outcome=arguments.get("outcome"),
            )
            return reflection.to_dict()

        elif name == "curate":
            from ctxf.engine.reflector import Reflection
            reflection = Reflection.from_dict(arguments["reflection"])
            cur_llm = create_llm(settings, "curator")
            curator = Curator(cur_llm, sections=pb.sections)
            deltas = await curator.curate(reflection)

            result_deltas = [d.to_dict() for d in deltas]

            if arguments.get("auto_apply", True) and deltas:
                store.apply_deltas(pb.id, deltas)

            return {"deltas": result_deltas, "applied": arguments.get("auto_apply", True)}

        elif name == "evolve":
            embedder = Embedder(
                base_url=settings.effective_embed_base_url,
                api_key=settings.effective_embed_api_key,
                model=settings.embed_model,
                dim=settings.embed_dim,
            ) if settings.embeddings_enabled else None

            retriever = HybridRetriever(
                embedder=embedder,
                top_k=settings.retrieval_top_k,
                alpha=settings.hybrid_alpha,
            )
            results = await retriever.retrieve(arguments["task"], pb.active_bullets())
            retrieved = [b for b, _ in results]

            gen_llm = create_llm(settings, "generator")
            generator = Generator(gen_llm)
            gen_result = await generator.execute(arguments["task"], retrieved)

            refl_llm = create_llm(settings, "reflector")
            reflector = Reflector(refl_llm)
            reflection = await reflector.reflect(
                task=arguments["task"],
                response=gen_result.response,
                outcome=arguments.get("outcome"),
                bullets_used=[b.to_dict() for b in retrieved],
            )

            cur_llm = create_llm(settings, "curator")
            curator = Curator(cur_llm, sections=pb.sections)
            deltas = await curator.curate(reflection)

            applied = False
            if arguments.get("auto_apply", True) and deltas:
                store.apply_deltas(pb.id, deltas)
                applied = True

            return {
                "response": gen_result.response,
                "reflection": reflection.to_dict(),
                "deltas": [d.to_dict() for d in deltas],
                "applied": applied,
            }

        else:
            return {"error": f"Unknown tool: {name}"}

    except Exception as e:
        logger.exception("Tool call failed")
        return {"error": str(e)}
    finally:
        store.close()


async def _process_message(msg: dict) -> str | None:
    """Process a single JSON-RPC message."""
    method = msg.get("method", "")
    msg_id = msg.get("id")
    params = msg.get("params", {})

    if method == "initialize":
        return _json_rpc_response(msg_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "context-forge", "version": "0.1.0"},
        })

    elif method == "notifications/initialized":
        return None  # No response for notifications

    elif method == "tools/list":
        return _json_rpc_response(msg_id, {"tools": TOOLS})

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        result = await _handle_tool_call(tool_name, arguments)
        return _json_rpc_response(msg_id, {
            "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
        })

    else:
        return _json_rpc_error(msg_id, -32601, f"Method not found: {method}")


async def run_mcp_server() -> None:
    """Run the MCP server reading from stdin, writing to stdout."""
    logger.info("Starting context-forge MCP server")
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, asyncio.get_event_loop())

    while True:
        try:
            line = await reader.readline()
            if not line:
                break
            line = line.decode("utf-8").strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                sys.stderr.write(f"Invalid JSON: {line}\n")
                continue

            response = await _process_message(msg)
            if response:
                writer.write(response.encode("utf-8"))
                await writer.drain()

        except Exception as e:
            logger.exception("Error processing message")
            sys.stderr.write(f"Error: {e}\n")
