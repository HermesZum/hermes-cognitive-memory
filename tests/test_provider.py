"""Tests for the CognitiveMemoryProvider plugin interface."""

import json
import pytest
from unittest.mock import MagicMock, patch

from cognitive_memory import CognitiveMemoryProvider
from cognitive_memory.decay import DecayParams


@pytest.fixture
def provider(tmp_path):
    """Create a provider with a temp directory."""
    p = CognitiveMemoryProvider()
    p.initialize(
        session_id="test-session",
        hermes_home=str(tmp_path),
        agent_context="primary",
    )
    yield p
    p.shutdown()


class TestProviderBasics:
    def test_name(self):
        p = CognitiveMemoryProvider()
        assert p.name == "cognitive"

    def test_is_available(self):
        p = CognitiveMemoryProvider()
        assert p.is_available() is True

    def test_initialize_creates_store(self, tmp_path):
        p = CognitiveMemoryProvider()
        p.initialize(
            session_id="test",
            hermes_home=str(tmp_path),
            agent_context="primary",
        )
        assert p._store is not None
        assert p._initialized is True
        p.shutdown()

    def test_shutdown_cleans_up(self, tmp_path):
        p = CognitiveMemoryProvider()
        p.initialize(
            session_id="test",
            hermes_home=str(tmp_path),
        )
        p.shutdown()
        assert p._store is None
        assert p._initialized is False


class TestSystemPrompt:
    def test_system_prompt_block_disabled(self, provider):
        """system_prompt_block() returns '' — markdown injection is disabled
        (the agent gets memories via inline prefetch, not the system prompt)."""
        provider._store.add("memory", "test entry")
        block = provider.system_prompt_block()
        assert block == ""


class TestPrefetch:
    def test_returns_empty_for_no_store(self):
        p = CognitiveMemoryProvider()
        assert p.prefetch("test query") == ""

    def test_returns_empty_for_trivial_prompt(self, provider):
        result = provider.prefetch("ok")
        assert result == ""

    def test_returns_empty_for_no_matches(self, provider):
        result = provider.prefetch("nonexistent topic")
        assert result == ""

    def test_returns_formatted_context(self, provider):
        provider._store.add("memory", "User prefers Python", origin="user_preference")
        result = provider.prefetch("Python")
        # Plugin returns the raw recalled text (the core wraps it in <memory-context>).
        assert "User prefers Python" in result
        assert "MEMORY" in result

    def test_includes_importance_bar(self, provider):
        provider._store.add("memory", "important fact", origin="user_correction")
        result = provider.prefetch("important")
        assert "▓" in result  # importance bar


class TestOnMemoryWrite:
    def test_add_mirrors_to_store(self, provider):
        provider.on_memory_write(
            action="add",
            target="memory",
            content="test memory content",
        )
        assert provider._store.count() == 1

    def test_replace_mirrors_as_new_add(self, provider):
        provider.on_memory_write(
            action="replace",
            target="memory",
            content="replaced content",
        )
        assert provider._store.count() == 1

    def test_remove_by_content(self, provider):
        provider.on_memory_write("add", "memory", "deletable content here")
        assert provider._store.count() == 1
        provider.on_memory_write("remove", "memory", "deletable content here")
        assert provider._store.count() == 0

    def test_user_correction_gets_high_importance(self, provider):
        provider.on_memory_write(
            "replace", "memory", "corrected fact",
        )
        all_mems = provider._store.get_all()
        assert len(all_mems) == 1
        assert all_mems[0]["importance"] >= 0.9
        assert all_mems[0]["origin"] == "user_correction"


class TestToolSchemas:
    def test_returns_five_tools(self, provider):
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 5
        names = [s["name"] for s in schemas]
        assert "cognitive_search" in names
        assert "cognitive_stats" in names
        assert "cognitive_remember" in names
        assert "cognitive_forget" in names
        assert "cognitive_sync_memory" in names


class TestHandleToolCall:
    def test_search_returns_results(self, provider):
        provider._store.add("memory", "searchable content about Python")
        result = provider.handle_tool_call("cognitive_search", {"query": "Python"})
        data = json.loads(result)
        assert data["count"] >= 1
        assert "Python" in data["memories"][0]["content"]

    def test_stats_returns_counts(self, provider):
        provider._store.add("memory", "mem1")
        provider._store.add("user", "user1")
        result = provider.handle_tool_call("cognitive_stats", {})
        data = json.loads(result)
        assert data["total_memories"] == 2
        assert data["memory_store"] == 1
        assert data["user_profile"] == 1

    def test_remember_stores_memory(self, provider):
        result = provider.handle_tool_call(
            "cognitive_remember",
            {"content": "remembered fact", "target": "memory", "origin": "user_preference"},
        )
        data = json.loads(result)
        assert data["status"] == "stored"
        assert provider._store.count() == 1

    def test_forget_deletes_memory(self, provider):
        mem_id = provider._store.add("memory", "will be forgotten")
        result = provider.handle_tool_call(
            "cognitive_forget",
            {"memory_id": mem_id},
        )
        data = json.loads(result)
        assert data["status"] == "deleted"
        assert provider._store.count() == 0

    def test_unknown_tool_returns_error(self, provider):
        result = provider.handle_tool_call("unknown_tool", {})
        data = json.loads(result)
        assert "error" in data


class TestSessionLifecycle:
    def test_on_session_end_prunes(self, provider):
        # Add a memory and force it below decay floor
        mem_id = provider._store.add("memory", "prunable", origin="agent_inference")
        provider._store._conn.execute(
            "UPDATE memories SET importance = 0.01 WHERE id = ?", (mem_id,)
        )
        provider._store._conn.commit()

        provider.on_session_end([])
        assert provider._store.count() == 0

    def test_on_session_switch_updates_session_id(self, provider):
        provider.on_session_switch("new-session", reset=True)
        assert provider._session_id == "new-session"

    def test_sync_turn_does_not_modify_importance(self, provider):
        """sync_turn no longer modifies stored importance — decay is computed on the fly."""
        mem_id = provider._store.add("memory", "decaying memory", origin="user_preference")
        # Set last_access to 1 day ago
        import time
        provider._store._conn.execute(
            "UPDATE memories SET last_access = ? WHERE id = ?",
            (time.time() - 86400, mem_id),
        )
        provider._store._conn.commit()

        before = provider._store.get(mem_id)["importance"]
        provider.sync_turn("user msg", "assistant msg")
        after = provider._store.get(mem_id)["importance"]
        # Stored importance should NOT change — decay is computed at retrieval time
        assert after == before