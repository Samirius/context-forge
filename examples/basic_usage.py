"""Basic usage example for context-forge."""

import json
from ctxf.store.sqlite_store import SqliteStore
from ctxf.engine.reflector import Reflector
from ctxf.engine.curator import Curator
from ctxf.engine.refiner import Refiner
from ctxf.retrieval.hybrid import HybridRetriever
from ctxf.llm.openai_compat import OpenAICompatProvider
from ctxf.config import get_settings


def main():
    settings = get_settings()

    # 1. Set up LLM provider
    llm = OpenAICompatProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )

    # 2. Set up store
    store = SqliteStore(settings.db_path)

    # 3. Create a playbook
    playbook_id = store.create_playbook(
        task_name="my-project",
        sections=["strategies", "common_mistakes", "insights"],
    )
    print(f"Created playbook: {playbook_id}")

    # 4. Retrieve relevant context for a query
    retriever = HybridRetriever(store)
    bullets = retriever.retrieve("how to handle API errors", playbook_id, top_k=5)
    print(f"Retrieved {len(bullets)} relevant bullets:")
    for b in bullets:
        print(f"  [{b.id}] helpful={b.helpful} harmful={b.harmful} :: {b.content}")

    # 5. Analyze a task outcome
    reflector = Reflector(llm)
    task_doc = {
        "query": "Implement error handling for the payment API",
        "response": "Added try-catch blocks and retry logic...",
        "outcome": "success",
        "test_output": "All 12 tests passed",
        "retrieved_bullets": [b.id for b in bullets],
    }
    reflection = reflector.reflect(json.dumps(task_doc))
    print(f"\nReflection: {reflection.summary}")
    for insight in reflection.insights:
        print(f"  - {insight.type}: {insight.content}")

    # 6. Convert insights to deltas
    curator = Curator(llm)
    playbook = store.get_playbook(playbook_id)
    deltas = curator.curate(reflection, playbook)
    print(f"\nGenerated {len(deltas)} deltas:")
    for d in deltas:
        print(f"  {d.op.value}: {d.reason}")

    # 7. Apply deltas
    store.apply_deltas(playbook_id, deltas)
    print("\nDeltas applied!")

    # 8. Run refiner periodically
    refiner = Refiner(llm, store)
    stats = refiner.refine(playbook_id, threshold=0.85)
    print(f"Refinement: deduped={stats.deduped}, pruned={stats.pruned}, merged={stats.merged}")

    # 9. View final stats
    final = store.get_playbook(playbook_id)
    print(f"\nPlaybook now has {len(final.bullets)} bullets across {len(final.sections)} sections")


if __name__ == "__main__":
    main()
