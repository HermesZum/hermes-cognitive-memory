"""Retrieval evaluation harness for the selective prefetch.

Why this exists
---------------
The selective prefetch (MemoryStore.search) had TWO silent regressions:
  1. Phrase-match bug — _sanitize_fts_query wrapped the whole query in FTS5
     quotes, so natural multi-word queries returned 0 selective memories.
  2. Empty-query crash — _importance_based_retrieval returned a bare list but
     _handle_search unpacked it as a tuple -> ValueError.

Both would have been caught automatically by a retrieval eval. This module
establishes a deterministic corpus + query/expected-id pairs and asserts the
invariants that MUST hold. Run it after ANY change to search/ranking.

Metrics
-------
  precision@k : fraction of the top-k selective results that are "relevant"
                (i.e. in the expected set for that query)
  recall@k    : fraction of the expected set that appears in the top-k
  MRR         : mean reciprocal rank of the first expected memory

Usage
-----
  pytest tests/test_retrieval_eval.py            # CI / assertions
  python tests/test_retrieval_eval.py            # print metrics table

NOTE: corpus uses FICTIONAL placeholders (e.g. host 'examplehost.example',
user 'fakeuser') per the project's test-fixture sanitization rule. Do not
put real environment data in this file.
"""

import sys
import math
import tempfile
from pathlib import Path

import pytest

from cognitive_memory.decay import DecayParams
from cognitive_memory.store import MemoryStore


# --------------------------------------------------------------------------
# Deterministic corpus — FICTIONAL placeholders only. (See module docstring.)
# Each entry: (id, target, content, importance, confidence, pinned, critical)
# IDs are short so they're easy to reference in eval cases.
# --------------------------------------------------------------------------
CORPUS = [
    ("mem_security_rules", "memory",
     "SECURITY RULE: never commit real environment data to public GitHub repos",
     0.95, 1.0, True, True),
    ("mem_git_branch", "memory",
     "Git rule: never assume target branch is main; inspect current branch first",
     0.95, 1.0, True, True),
    ("mem_obsidian_vault", "memory",
     "Obsidian vault: /root/vault, private GitHub fakeorg/vault-brain",
     0.7, 0.9, False, False),
    ("mem_vault_brain", "memory",
     "Vault brain behavioural instruction: check the vault first for FX/infra",
     0.8, 0.9, True, False),
    ("mem_hermes_source", "memory",
     "Hermes source /usr/local/lib/hermes-agent: origin AcmeAI, fork FakeUser",
     0.75, 0.85, False, False),
    ("mem_ssh_user", "memory",
     "SSH user on examplehost.example is fakeuser — never pkill -f hermes (kills SSH)",
     0.85, 0.9, True, True),
    ("mem_fx_demo", "memory",
     "FX (TraderX): demo-first. Graduate to live after >=20 consistent demo trades",
     0.6, 0.8, False, False),
    ("mem_markdown_style", "user",
     "Always use markdown formatting in responses: code blocks with backticks",
     0.5, 0.9, False, False),
    ("mem_no_live_changes", "user",
     "Do NOT make live system changes to the platform Hermes runs on",
     0.55, 0.9, False, False),
    ("mem_pkill_rule", "memory",
     "Never use pkill -f hermes — matches the SSH session; use fuser -k <port>/tcp",
     0.8, 0.9, True, True),
    ("mem_definition_done", "memory",
     "DEFINITION OF DONE: 8 verification gates before claiming done",
     0.9, 1.0, True, True),
    ("mem_model_fallback", "memory",
     "Model fallback chain: glm-5.2:cloud primary, tencent/hy3 fallback",
     0.65, 0.8, False, False),
]


def _seed(store: MemoryStore) -> dict:
    """Seed the corpus. `critical`/`last_access` aren't add() kwargs, so we
    set them via direct UPDATE after insertion (mirrors test_store.py pattern).
    Returns {corpus_id: actual_stored_id} so eval cases can reference real ids.
    """
    id_map = {}
    now = math.floor(__import__("time").time())
    for mid, target, content, imp, conf, pinned, critical in CORPUS:
        mem_id = store.add(
            target=target,
            content=content,
            origin="eval_seed",
            importance=imp,
            confidence=conf,
            pinned=pinned,
        )
        store._conn.execute(
            "UPDATE memories SET critical = ?, created_at = ?, last_access = ? WHERE id = ?",
            (1 if critical else 0, now, now, mem_id),
        )
        store._conn.commit()
        id_map[mid] = mem_id
    return id_map


@pytest.fixture
def store(tmp_path):
    db = tmp_path / "eval_memory.db"
    s = MemoryStore(db, DecayParams())
    s.connect()
    id_map = _seed(s)
    yield s, id_map
    s.close()


# --------------------------------------------------------------------------
# Eval cases — (query, {expected_relevant_ids}, k)
# These encode the behavior we REQUIRE. If a retrieval change breaks any of
# these, the test fails loudly instead of shipping a silent regression.
# --------------------------------------------------------------------------
EVAL_CASES = [
    # Natural multi-word query MUST retrieve the right memory (regression #1 guard)
    ("What are the security rules for this system?",
     {"mem_security_rules", "mem_git_branch", "mem_ssh_user", "mem_pkill_rule"}, 5),
    # Token overlap must work (not phrase-only)
    ("Obsidian vault location", {"mem_obsidian_vault", "mem_vault_brain"}, 5),
    # Specific term should outrank ubiquitous token
    ("pkill hermes command", {"mem_pkill_rule", "mem_ssh_user"}, 5),
    # Hermes source query
    ("where is the hermes source code", {"mem_hermes_source"}, 5),
    # FX
    ("FX demo trading rules", {"mem_fx_demo"}, 5),
    # Definition of done
    ("definition of done gates", {"mem_definition_done"}, 5),
    # Genuinely-unrelated queries -> no selective (only critical tier handles it)
    ("how to tune a guitar", set(), 5),
    ("best recipe for bacalhau", set(), 5),
]

# Semantic (paraphrase) eval cases — these are the queries lexical-only
# retrieval MISSes. They only pass when the dense/semantic path is active
# (Ollama nomic-embed-text available). Guarded by SEMANTIC_AVAILABLE so the
# harness stays green in CI without a local embedding model.
SEMANTIC_CASES = [
    # "never assume target branch is main" — wording differs from corpus
    ("how do I push code safely without breaking things?",
     {"mem_git_branch"}, 5),
    # "never use pkill -f hermes (kills SSH)" — phrased as a question
    ("is it safe to kill the hermes process from the terminal?",
     {"mem_ssh_user", "mem_pkill_rule"}, 5),
    # "never commit real environment data to public GitHub" — paraphrase
    ("what must never go into a public git repository?",
     {"mem_security_rules"}, 5),
]


def _ollama_available() -> bool:
    """True if Ollama is reachable AND nomic-embed-text can serve embeddings."""
    try:
        from cognitive_memory import embeddings as _emb
        backend = _emb.OllamaEmbeddingBackend()
        if not backend.available:
            return False
        return backend.embed("probe") is not None
    except Exception:  # noqa: BLE001
        return False


SEMANTIC_AVAILABLE = _ollama_available()


def _relevant_in_topk(store, query, k):
    """Return selective ids in top-k (list)."""
    results, _critical = store.search(query, limit=k)
    return [m["id"] for m, _ in results]


def _relevant_combined(store, query, k):
    """Return combined memory ids (selective top-k + critical tier).

    The agent's <memory-context> shows BOTH the selective pool AND the
    always-on critical section, so relevance is judged on the union.
    Criticals are always injected regardless of query, so they count as
    "retrieved" for any query.
    """
    results, critical = store.search(query, limit=k)
    ids = [m["id"] for m, _ in results]
    ids += [m["id"] for m in critical]
    return ids


# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------
def _precision_at_k(topk_ids, expected, k):
    """Fraction of retrieved memories that are relevant.

    Uses the FULL retrieved set (not sliced to k) for consistency with
    _recall_at_k: the combined retrieval (selective + critical) can exceed k
    and the agent sees ALL of it, so precision must measure the entire shown
    set. Returns 1.0 if nothing is retrieved.
    """
    if not topk_ids:
        return 1.0
    hits = sum(1 for i in topk_ids if i in expected)
    return hits / len(topk_ids)


def _recall_at_k(topk_ids, expected, k):
    """Fraction of expected memories present in the retrieved set.

    Uses the FULL retrieved set (not sliced to k) because the combined
    retrieval (selective + critical) can legitimately exceed k — the agent
    sees all of it. Recall measures coverage, not rank-position-within-k.
    """
    if not expected:
        # Unrelated query: recall is vacuously 1.0 if nothing retrieved
        return 1.0 if not topk_ids else 0.0
    hits = sum(1 for i in topk_ids if i in expected)
    return hits / len(expected)


def _mrr(topk_ids, expected):
    if not expected:
        return 1.0  # no relevant item expected -> perfect
    for rank, i in enumerate(topk_ids, start=1):
        if i in expected:
            return 1.0 / rank
    return 0.0


# --------------------------------------------------------------------------
# Tests (assertions — these are the regression guards)
# --------------------------------------------------------------------------
class TestRetrievalEval:
    def test_empty_query_does_not_crash(self, store):
        """Regression #2: empty query must return a tuple, not crash."""
        s, _ = store
        results, critical = s.search("", limit=5)
        assert isinstance(results, list)
        assert isinstance(critical, list)

    def test_natural_query_retrieves_security_memory(self, store):
        """Regression #1: natural multi-word query must surface security rules
        (in either the selective pool or the critical safety tier)."""
        s, id_map = store
        combined = _relevant_combined(s, "What are the security rules for this system?", 5)
        assert id_map["mem_security_rules"] in combined

    def test_unrelated_query_has_no_selective_noise(self, store):
        """'weather' must not inject unrelated memories into selective."""
        s, _ = store
        topk = _relevant_in_topk(s, "weather in Lisbon", 5)
        assert topk == []

    def test_unrelated_queries_have_no_selective_noise(self, store):
        """Genuinely-unrelated queries must not surface selective memories.

        Under hybrid retrieval, in-domain queries legitimately return
        semantically-related memories. Only truly-unrelated queries (guitar,
        recipe) should yield an empty selective pool. This guards against the
        semantic floor being too low (which would let ~0.38-0.40 cosine
        'unrelated' matches pollute retrieval).
        """
        s, _ = store
        s.configure_embeddings(enabled=True)
        for q in ("how to tune a guitar", "best recipe for bacalhau"):
            topk = _relevant_in_topk(s, q, 5)
            assert topk == [], f"unrelated query {q!r} returned selective: {topk}"

    def test_stopword_query_no_lexical_noise(self, store):
        """Stopword-heavy query must not surface 'the'-only lexical noise.

        Verified in lexical-only mode (semantic off) so we isolate the
        original regression: the stopword filter must drop 'the' so it can't
        match 4 memories and flood the budget.
        """
        s, _ = store
        s.configure_embeddings(enabled=False)
        topk = _relevant_in_topk(s, "audit second pass the selective prefetch", 5)
        assert topk == []

    def test_criticals_never_leak_into_selective(self, store):
        """Critical isolation invariant (architectural)."""
        s, _ = store
        for query, _, _ in EVAL_CASES:
            results, critical = s.search(query, limit=15)
            sel_ids = {m["id"] for m, _ in results}
            crit_ids = {m["id"] for m in critical}
            assert not (sel_ids & crit_ids), f"critical leaked for {query!r}"

    def test_selective_never_exceeds_cap(self, store):
        s, _ = store
        results, _ = s.search("hermes source code branch", limit=15)
        assert len(results) <= 15

    def test_all_eval_cases_meet_minimum_recall(self, store):
        """Every eval case must achieve recall >= 0.5 on the COMBINED retrieval
        (selective + critical tier), or be empty-as-designed for unrelated queries."""
        s, id_map = store
        for query, expected, k in EVAL_CASES:
            expected_real = {id_map[e] for e in expected}
            topk = _relevant_combined(s, query, k)
            r = _recall_at_k(topk, expected_real, k)
            if expected:
                assert r >= 0.5, f"recall {r:.2f} < 0.5 for {query!r} (got {topk})"

    @pytest.mark.skipif(not SEMANTIC_AVAILABLE,
                        reason="Ollama nomic-embed-text not available")
    def test_semantic_paraphrase_recall(self, store):
        """Paraphrase queries must surface their target via the dense path.

        This is the regression guard for hybrid retrieval: lexical-only misses
        these (different wording), so if it passes, the semantic fusion works.
        Skipped when the embedding backend is unreachable (CI without Ollama).
        """
        s, id_map = store
        s.configure_embeddings(enabled=True)  # ensure hybrid is on
        for query, expected, k in SEMANTIC_CASES:
            expected_real = {id_map[e] for e in expected}
            topk = _relevant_combined(s, query, k)
            r = _recall_at_k(topk, expected_real, k)
            assert r >= 0.5, f"semantic recall {r:.2f} < 0.5 for {query!r}"


# --------------------------------------------------------------------------
# CLI metric report (python tests/test_retrieval_eval.py)
# --------------------------------------------------------------------------
def _build_store_for_cli():
    """Build a CLI store and return (store, id_map)."""
    tmp = Path(tempfile.mkdtemp())
    s = MemoryStore(tmp / "eval_memory.db", DecayParams())
    s.connect()
    id_map = _seed(s)
    return s, id_map


def main():
    store, id_map = _build_store_for_cli()
    print(f"{'QUERY':52} {'P@5':>6} {'R@5':>6} {'MRR':>6}  selective")
    print("-" * 90)
    tot_p = tot_r = tot_m = 0.0
    n = 0
    for query, expected, k in EVAL_CASES:
        expected_real = {id_map[e] for e in expected}
        topk = _relevant_combined(store, query, k)
        p = _precision_at_k(topk, expected_real, k)
        r = _recall_at_k(topk, expected_real, k)
        m = _mrr(topk, expected_real)
        tot_p += p; tot_r += r; tot_m += m; n += 1
        print(f"{query[:52]:52} {p:6.2f} {r:6.2f} {m:6.2f}  {len(topk)}")
    print("-" * 90)
    print(f"{'MACRO-AVG':52} {tot_p/n:6.2f} {tot_r/n:6.2f} {tot_m/n:6.2f}")
    store.close()


if __name__ == "__main__":
    sys.exit(main())
