"""context-forge CLI — command-line interface for Agentic Context Engineering."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click

from ctxf import __version__
from ctxf.config import get_settings


@click.group()
@click.version_option(__version__, prog_name="ctxf")
def cli():
    """context-forge (ctxf) — Self-hostable Agentic Context Engineering.

    Evolve your AI playbook automatically through the ACE loop:
    retrieve → generate → reflect → curate → apply.
    """
    pass


# --- init ---

@cli.command()
@click.option("--name", default="default", help="Playbook name")
@click.option("--db", "db_path", default=None, help="Database path")
def init(name: str, db_path: Optional[str]):
    """Create a new playbook with default sections."""
    from ctxf.store.sqlite_store import SqliteStore

    settings = get_settings()
    store = SqliteStore(db_path or settings.db_path)

    try:
        pb = store.create_playbook(name=name)
        click.echo(f"Created playbook: {pb.name} (id: {pb.id})")
        click.echo(f"Version: {pb.version}")
        click.echo(f"Sections: {', '.join(pb.sections)}")
        click.echo(f"Database: {store.db_path}")
    finally:
        store.close()


# --- retrieve ---

@cli.command()
@click.argument("query")
@click.option("--playbook-id", default=None, help="Playbook ID")
@click.option("--top-k", default=10, help="Number of results")
@click.option("--db", "db_path", default=None, help="Database path")
def retrieve(query: str, playbook_id: Optional[str], top_k: int, db_path: Optional[str]):
    """Search playbook for relevant bullets."""
    from ctxf.store.sqlite_store import SqliteStore
    from ctxf.retrieval.hybrid import HybridRetriever
    from ctxf.retrieval.embedder import Embedder

    settings = get_settings()
    store = SqliteStore(db_path or settings.db_path)

    try:
        pb = store.get_playbook(playbook_id)
        if not pb:
            click.echo("No playbook found. Run 'ctxf init' first.", err=True)
            sys.exit(1)

        embedder = None
        if settings.embeddings_enabled:
            embedder = Embedder(
                base_url=settings.effective_embed_base_url,
                api_key=settings.effective_embed_api_key,
                model=settings.embed_model,
                dim=settings.embed_dim,
            )

        retriever = HybridRetriever(
            embedder=embedder,
            top_k=top_k,
            alpha=settings.hybrid_alpha,
        )

        results = asyncio.run(retriever.retrieve(query, pb.active_bullets()))

        if not results:
            click.echo("No results found.")
            return

        for bullet, score in results:
            click.echo(f"[{bullet.id}] ({bullet.section}) score={score:.4f}")
            click.echo(f"  {bullet.content}")
            click.echo()
    finally:
        store.close()


# --- reflect ---

@cli.command()
@click.option("--doc", "doc_path", type=click.Path(exists=True), help="Path to task JSON file")
@click.option("--task", default=None, help="Task description (inline)")
@click.option("--response", default=None, help="Response text (inline)")
@click.option("--outcome", default=None, help="Outcome feedback (inline)")
@click.option("--output", "-o", default=None, help="Output file for reflection JSON")
@click.option("--db", "db_path", default=None, help="Database path")
def reflect(doc_path, task, response, outcome, output, db_path):
    """Analyze task outcome and extract insights."""
    from ctxf.store.sqlite_store import SqliteStore
    from ctxf.llm.factory import create_llm
    from ctxf.engine.reflector import Reflector

    if doc_path:
        with open(doc_path) as f:
            doc = json.load(f)
        task = doc.get("task", task or "")
        response = doc.get("response", response or "")
        outcome = doc.get("outcome", outcome)

    if not task:
        click.echo("Error: --task or --doc with 'task' field required", err=True)
        sys.exit(1)

    settings = get_settings()
    store = SqliteStore(db_path or settings.db_path)

    try:
        # Get bullets used if playbook exists
        bullets_used = None
        pb = store.get_playbook()
        if pb:
            bullets_used = [b.to_dict() for b in pb.active_bullets()[:20]]

        llm = create_llm(settings, "reflector")
        reflector = Reflector(llm)
        reflection = asyncio.run(reflector.reflect(
            task=task,
            response=response or "",
            outcome=outcome,
            bullets_used=bullets_used,
        ))

        result = reflection.to_dict()
        output_json = json.dumps(result, indent=2)

        if output:
            Path(output).write_text(output_json)
            click.echo(f"Reflection saved to {output}")
        else:
            click.echo(output_json)
    finally:
        store.close()


# --- curate ---

@cli.command()
@click.option("--reflection", "reflection_path", type=click.Path(exists=True), help="Path to reflection JSON")
@click.option("--output", "-o", default=None, help="Output file for deltas JSON")
@click.option("--db", "db_path", default=None, help="Database path")
def curate(reflection_path, output, db_path):
    """Convert a reflection into delta operations."""
    from ctxf.store.sqlite_store import SqliteStore
    from ctxf.llm.factory import create_llm
    from ctxf.engine.reflector import Reflection
    from ctxf.engine.curator import Curator

    if not reflection_path:
        click.echo("Error: --reflection required", err=True)
        sys.exit(1)

    with open(reflection_path) as f:
        reflection_data = json.load(f)

    settings = get_settings()
    store = SqliteStore(db_path or settings.db_path)

    try:
        pb = store.get_playbook()
        sections = pb.sections if pb else None

        llm = create_llm(settings, "curator")
        curator = Curator(llm, sections=sections)
        reflection = Reflection.from_dict(reflection_data)
        deltas = asyncio.run(curator.curate(reflection))

        result = [d.to_dict() for d in deltas]
        output_json = json.dumps(result, indent=2)

        if output:
            Path(output).write_text(output_json)
            click.echo(f"Deltas saved to {output}")
        else:
            click.echo(output_json)
    finally:
        store.close()


# --- commit (apply deltas) ---

@cli.command()
@click.option("--delta", "delta_path", type=click.Path(exists=True), help="Path to delta JSON file")
@click.option("--playbook-id", default=None, help="Playbook ID")
@click.option("--db", "db_path", default=None, help="Database path")
def commit(delta_path, playbook_id, db_path):
    """Apply delta operations to the playbook."""
    from ctxf.store.sqlite_store import SqliteStore
    from ctxf.models.delta import Delta

    if not delta_path:
        click.echo("Error: --delta required", err=True)
        sys.exit(1)

    with open(delta_path) as f:
        delta_data = json.load(f)

    if isinstance(delta_data, dict):
        delta_data = [delta_data]

    deltas = [Delta.from_dict(d) for d in delta_data]

    settings = get_settings()
    store = SqliteStore(db_path or settings.db_path)

    try:
        pb = store.get_playbook(playbook_id)
        if not pb:
            click.echo("No playbook found. Run 'ctxf init' first.", err=True)
            sys.exit(1)

        results = store.apply_deltas(pb.id, deltas)
        click.echo(f"Applied {len([r for r in results if r])} deltas to playbook {pb.name}")
    finally:
        store.close()


# --- evolve (full loop) ---

@cli.command()
@click.option("--task", "task_path", type=click.Path(exists=True), help="Path to task JSON file")
@click.option("--task-text", default=None, help="Task description (inline)")
@click.option("--outcome", default=None, help="Outcome feedback")
@click.option("--auto-apply/--no-auto-apply", default=True, help="Auto-apply deltas")
@click.option("--db", "db_path", default=None, help="Database path")
def evolve(task_path, task_text, outcome, auto_apply, db_path):
    """Run the full ACE loop: reflect → curate → commit."""
    from ctxf.store.sqlite_store import SqliteStore
    from ctxf.retrieval.embedder import Embedder
    from ctxf.retrieval.hybrid import HybridRetriever
    from ctxf.engine.generator import Generator
    from ctxf.engine.reflector import Reflector
    from ctxf.engine.curator import Curator
    from ctxf.llm.factory import create_llm

    task = task_text
    if task_path:
        with open(task_path) as f:
            doc = json.load(f)
        task = doc.get("task", task)
        outcome = doc.get("outcome", outcome)

    if not task:
        click.echo("Error: --task-text or --task (file) required", err=True)
        sys.exit(1)

    settings = get_settings()
    store = SqliteStore(db_path or settings.db_path)

    try:
        pb = store.get_playbook()
        if not pb:
            click.echo("No playbook found. Run 'ctxf init' first.", err=True)
            sys.exit(1)

        # Retrieve
        click.echo("Retrieving relevant context...")
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
        results = asyncio.run(retriever.retrieve(task, pb.active_bullets()))
        retrieved = [b for b, _ in results]
        click.echo(f"  Found {len(retrieved)} relevant bullets")

        # Generate
        click.echo("Generating response...")
        gen_llm = create_llm(settings, "generator")
        generator = Generator(gen_llm)
        gen_result = asyncio.run(generator.execute(task, retrieved))
        click.echo(f"  Response generated ({len(gen_result.response)} chars)")

        # Reflect
        click.echo("Reflecting on execution...")
        refl_llm = create_llm(settings, "reflector")
        reflector = Reflector(refl_llm)
        reflection = asyncio.run(reflector.reflect(
            task=task,
            response=gen_result.response,
            outcome=outcome,
            bullets_used=[b.to_dict() for b in retrieved],
        ))
        click.echo(f"  Reflection: {reflection.summary[:100]}")

        # Curate
        click.echo("Curating deltas...")
        cur_llm = create_llm(settings, "curator")
        curator = Curator(cur_llm, sections=pb.sections)
        deltas = asyncio.run(curator.curate(reflection))
        click.echo(f"  Generated {len(deltas)} deltas")

        # Apply
        if auto_apply and deltas:
            click.echo("Applying deltas...")
            store.apply_deltas(pb.id, deltas)
            click.echo(f"  Applied {len(deltas)} deltas")
        else:
            click.echo("Skipping auto-apply (--no-auto-apply or no deltas)")

        click.echo("\n--- Response ---")
        click.echo(gen_result.response[:1000])
        if len(gen_result.response) > 1000:
            click.echo("... (truncated)")
    finally:
        store.close()


# --- serve ---

@cli.command()
@click.option("--host", default=None, help="Host to bind")
@click.option("--port", default=None, type=int, help="Port to bind")
@click.option("--db", "db_path", default=None, help="Database path")
def serve(host, port, db_path):
    """Start the REST API server."""
    import uvicorn
    from ctxf.api.server import create_app

    settings = get_settings()
    if db_path:
        import os
        os.environ["CTXF_DB_PATH"] = db_path

    app = create_app()
    uvicorn.run(
        app,
        host=host or settings.server_host,
        port=port or settings.server_port,
        log_level=settings.log_level.lower(),
    )


# --- mcp ---

@cli.command()
def mcp():
    """Start the MCP stdio server."""
    from ctxf.mcp.server import run_mcp_server
    asyncio.run(run_mcp_server())


# --- stats ---

@cli.command()
@click.option("--playbook-id", default=None, help="Playbook ID")
@click.option("--db", "db_path", default=None, help="Database path")
def stats(playbook_id, db_path):
    """Show playbook statistics."""
    from ctxf.store.sqlite_store import SqliteStore

    settings = get_settings()
    store = SqliteStore(db_path or settings.db_path)

    try:
        s = store.get_stats(playbook_id)
        if "error" in s:
            click.echo("No playbook found. Run 'ctxf init' first.", err=True)
            sys.exit(1)

        click.echo(f"Playbook: {s['name']} (id: {s['playbook_id'][:8]}...)")
        click.echo(f"Version: {s['version']}")
        click.echo(f"Active bullets: {s['active_bullets']}")
        click.echo(f"Deprecated bullets: {s['deprecated_bullets']}")
        click.echo(f"Total helpful votes: {s['total_helpful']}")
        click.echo(f"Total harmful votes: {s['total_harmful']}")
        click.echo()
        click.echo("By section:")
        for section, count in s["by_section"].items():
            click.echo(f"  {section}: {count}")
    finally:
        store.close()


if __name__ == "__main__":
    cli()
