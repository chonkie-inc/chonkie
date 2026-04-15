"""Tests to improve coverage for pipeline.py and registry.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chonkie.pipeline.component import ComponentType
from chonkie.pipeline.pipeline import Pipeline
from chonkie.pipeline.registry import (
    ComponentRegistry,
    _ComponentRegistry,
    pipeline_component,
)
from chonkie.types import Document

# ---------------------------------------------------------------------------
# Helpers: minimal stub classes for registry tests
# ---------------------------------------------------------------------------


class _StubFetcher:
    def fetch(self): ...


class _StubFetcherAlt:
    """Different class with same name intent, for conflict tests."""

    def fetch(self): ...


class _StubChef:
    def process(self, path): ...
    def parse(self, text): ...


class _StubChunker:
    def chunk(self, text): ...
    def chunk_document(self, doc): ...


class _StubRefinery:
    def refine(self, chunks): ...
    def refine_document(self, doc): ...


class _StubPorter:
    def export(self, chunks): ...


class _StubHandshake:
    def write(self, chunks): ...


# ===========================================================================
# Registry tests
# ===========================================================================


class TestRegistryRegister:
    """Tests for _ComponentRegistry.register."""

    def test_idempotent_registration(self):
        """Re-registering the same class with same name is a no-op."""
        reg = _ComponentRegistry()
        reg.register("StubFetcher", "stub", _StubFetcher, ComponentType.FETCHER)
        # Same call again — should not raise
        reg.register("StubFetcher", "stub", _StubFetcher, ComponentType.FETCHER)
        assert len(reg.list_components()) == 1

    def test_name_conflict_different_class(self):
        """Registering a different class under the same name raises ValueError."""
        reg = _ComponentRegistry()
        reg.register("StubFetcher", "stub", _StubFetcher, ComponentType.FETCHER)
        with pytest.raises(ValueError, match="already registered with different class"):
            reg.register("StubFetcher", "stub2", _StubFetcherAlt, ComponentType.FETCHER)

    def test_alias_conflict_same_type(self):
        """Two different components sharing an alias within the same type raises ValueError."""
        reg = _ComponentRegistry()
        reg.register("Fetcher1", "shared_alias", _StubFetcher, ComponentType.FETCHER)
        with pytest.raises(ValueError, match="Alias 'shared_alias' already used"):
            reg.register("Fetcher2", "shared_alias", _StubFetcherAlt, ComponentType.FETCHER)


class TestRegistryGetComponent:
    """Tests for _ComponentRegistry.get_component edge cases."""

    @pytest.fixture(autouse=True)
    def _fresh_registry(self):
        self.reg = _ComponentRegistry()
        self.reg.register("MyChunker", "mychunk", _StubChunker, ComponentType.CHUNKER)
        self.reg.register("MyFetcher", "myfetch", _StubFetcher, ComponentType.FETCHER)

    def test_scoped_alias_lookup(self):
        comp = self.reg.get_component("mychunk", ComponentType.CHUNKER)
        assert comp.name == "MyChunker"

    def test_direct_name_lookup(self):
        comp = self.reg.get_component("MyChunker")
        assert comp.alias == "mychunk"

    def test_name_lookup_with_wrong_type_raises(self):
        """Looking up by exact name but requesting a mismatched type."""
        with pytest.raises(ValueError, match="not a fetcher"):
            self.reg.get_component("MyChunker", ComponentType.FETCHER)

    def test_unscoped_alias_single_match(self):
        """Unscoped alias lookup with exactly one match succeeds."""
        comp = self.reg.get_component("mychunk")
        assert comp.name == "MyChunker"

    def test_ambiguous_alias_raises(self):
        """Same alias in two different types without specifying type raises."""
        # Register another component with the same alias but different type
        self.reg.register("MyRefinery", "shared", _StubRefinery, ComponentType.REFINERY)
        self.reg.register("MyPorter", "shared", _StubPorter, ComponentType.PORTER)
        with pytest.raises(ValueError, match="Ambiguous alias"):
            self.reg.get_component("shared")

    def test_not_found_raises(self):
        with pytest.raises(ValueError, match="Unknown component"):
            self.reg.get_component("nonexistent")


class TestRegistryListAndAliases:
    """Tests for list_components and get_aliases."""

    @pytest.fixture(autouse=True)
    def _fresh_registry(self):
        self.reg = _ComponentRegistry()
        self.reg.register("F1", "f1", _StubFetcher, ComponentType.FETCHER)
        self.reg.register("C1", "c1", _StubChunker, ComponentType.CHUNKER)
        self.reg.register("C2", "c2", _StubChef, ComponentType.CHEF)

    def test_list_components_all(self):
        assert len(self.reg.list_components()) == 3

    def test_list_components_by_type(self):
        result = self.reg.list_components(ComponentType.FETCHER)
        assert len(result) == 1
        assert result[0].name == "F1"

    def test_get_aliases_all(self):
        aliases = self.reg.get_aliases()
        assert set(aliases) == {"f1", "c1", "c2"}

    def test_get_aliases_by_type(self):
        aliases = self.reg.get_aliases(ComponentType.CHUNKER)
        assert aliases == ["c1"]


class TestRegistryIsRegistered:
    """Tests for is_registered."""

    @pytest.fixture(autouse=True)
    def _fresh_registry(self):
        self.reg = _ComponentRegistry()
        self.reg.register("MyFetcher", "myfetch", _StubFetcher, ComponentType.FETCHER)

    def test_by_name(self):
        assert self.reg.is_registered("MyFetcher") is True

    def test_by_alias(self):
        assert self.reg.is_registered("myfetch") is True

    def test_not_registered(self):
        assert self.reg.is_registered("nope") is False


class TestRegistryUnregister:
    """Tests for unregister."""

    def test_unregister_existing(self):
        reg = _ComponentRegistry()
        reg.register("MyFetcher", "myfetch", _StubFetcher, ComponentType.FETCHER)
        assert reg.is_registered("MyFetcher")
        reg.unregister("MyFetcher")
        assert not reg.is_registered("MyFetcher")

    def test_unregister_not_found_is_noop(self):
        reg = _ComponentRegistry()
        reg.unregister("nonexistent")  # Should not raise


class TestRegistryClear:
    """Tests for clear."""

    def test_clear_removes_all(self):
        reg = _ComponentRegistry()
        reg.register("F1", "f1", _StubFetcher, ComponentType.FETCHER)
        reg.register("C1", "c1", _StubChunker, ComponentType.CHUNKER)
        assert len(reg.list_components()) == 2
        reg.clear()
        assert len(reg.list_components()) == 0


class TestRegistryGetHandshake:
    """Test get_handshake reaches the handshake code path."""

    def test_get_handshake(self):
        reg = _ComponentRegistry()
        reg.register("HS1", "hs1", _StubHandshake, ComponentType.HANDSHAKE)
        comp = reg.get_handshake("hs1")
        assert comp.name == "HS1"
        assert comp.component_type == ComponentType.HANDSHAKE


class TestPipelineComponentDecorator:
    """Tests for the pipeline_component decorator and specialized decorators."""

    def test_missing_methods_raises(self):
        """Class without required methods cannot be registered."""

        class BadFetcher:
            pass  # No fetch() method

        with pytest.raises(ValueError, match="must implement"):
            pipeline_component("bad", ComponentType.FETCHER)(BadFetcher)

    def test_decorator_adds_metadata(self):
        """Decorator sets _pipeline_component_info on the class."""
        reg_backup_components = dict(ComponentRegistry._components)
        reg_backup_aliases = dict(ComponentRegistry._aliases)
        reg_backup_types = {k: list(v) for k, v in ComponentRegistry._component_types.items()}

        try:

            @pipeline_component("test_deco_fetcher", ComponentType.FETCHER)
            class DecoTestFetcher:
                def fetch(self): ...

            assert hasattr(DecoTestFetcher, "_pipeline_component_info")
            assert DecoTestFetcher._pipeline_component_info["alias"] == "test_deco_fetcher"
        finally:
            # Restore global registry
            ComponentRegistry._components = reg_backup_components
            ComponentRegistry._aliases = reg_backup_aliases
            ComponentRegistry._component_types = reg_backup_types


# ===========================================================================
# Pipeline tests
# ===========================================================================


class TestPipelineFromConfig:
    """Tests for Pipeline.from_config."""

    def test_from_config_tuple_3_elements(self):
        """Tuple with (type, component, kwargs)."""
        pipeline = Pipeline.from_config([
            ("chunk", "recursive", {"chunk_size": 512}),
        ])
        assert len(pipeline._steps) == 1
        assert pipeline._steps[0]["type"] == "chunk"

    def test_from_config_tuple_2_elements(self):
        """Tuple with (type, component) — no kwargs."""
        pipeline = Pipeline.from_config([
            ("chunk", "recursive"),
        ])
        assert len(pipeline._steps) == 1

    def test_from_config_dict_format(self):
        """Dict with type, component, and extra kwargs."""
        pipeline = Pipeline.from_config([
            {"type": "chunk", "component": "recursive", "chunk_size": 256},
        ])
        assert len(pipeline._steps) == 1
        assert pipeline._steps[0]["kwargs"]["chunk_size"] == 256

    def test_from_config_json_file(self, tmp_path):
        """Load config from a JSON file path."""
        config = [
            {"type": "chunk", "component": "token", "chunk_size": 100},
        ]
        config_path = tmp_path / "pipeline.json"
        config_path.write_text(json.dumps(config))

        pipeline = Pipeline.from_config(str(config_path))
        assert len(pipeline._steps) == 1

    def test_from_config_invalid_tuple_length(self):
        with pytest.raises(ValueError, match="Tuple must have 2 or 3 elements"):
            Pipeline.from_config([("chunk",)])

    def test_from_config_invalid_step_type_in_tuple(self):
        with pytest.raises(ValueError, match="Unknown step type"):
            Pipeline.from_config([("bogus_type", "recursive")])

    def test_from_config_dict_missing_keys(self):
        with pytest.raises(ValueError, match="must have 'type' and 'component' keys"):
            Pipeline.from_config([{"type": "chunk"}])  # Missing 'component'

    def test_from_config_invalid_step_format(self):
        with pytest.raises(ValueError, match="Step must be tuple or dict"):
            Pipeline.from_config([42])

    def test_from_config_all_step_types(self, tmp_path):
        """Exercise fetch, process, chunk, refine, export, write step types."""
        config = [
            ("fetch", "file"),
            ("process", "text"),
            ("chunk", "recursive"),
            ("refine", "overlap"),
            ("export", "json"),
        ]
        pipeline = Pipeline.from_config(config)
        step_types = [s["type"] for s in pipeline._steps]
        assert step_types == ["fetch", "process", "chunk", "refine", "export"]

    def test_from_config_write_step(self):
        """The 'write' step type maps to store_in."""
        # Use a handshake that exists in the registry
        config = [
            ("chunk", "recursive"),
            ("write", "chroma"),
        ]
        pipeline = Pipeline.from_config(config)
        assert any(s["type"] == "write" for s in pipeline._steps)


class TestPipelineFromRecipe:
    """Tests for Pipeline.from_recipe error paths."""

    def test_from_recipe_empty_steps_raises(self):
        """Recipe with no steps raises ValueError."""
        with patch("chonkie.pipeline.pipeline.Hubbie") as MockHubbie:
            mock_instance = MockHubbie.return_value
            mock_instance.get_pipeline_recipe.return_value = {"steps": []}
            with pytest.raises(ValueError, match="no steps defined"):
                Pipeline.from_recipe("empty_recipe")

    def test_from_recipe_with_steps(self):
        """Recipe with valid steps creates pipeline."""
        with patch("chonkie.pipeline.pipeline.Hubbie") as MockHubbie:
            mock_instance = MockHubbie.return_value
            mock_instance.get_pipeline_recipe.return_value = {
                "steps": [
                    {"type": "chunk", "component": "recursive", "chunk_size": 512},
                ]
            }
            pipeline = Pipeline.from_recipe("test_recipe")
            assert len(pipeline._steps) == 1


class TestPipelineExportAndStore:
    """Tests for export_with and store_in method chaining."""

    def test_export_with_returns_pipeline(self):
        pipeline = Pipeline().export_with("json")
        assert isinstance(pipeline, Pipeline)
        assert pipeline._steps[-1]["type"] == "export"

    def test_store_in_returns_pipeline(self):
        pipeline = Pipeline().store_in("chroma")
        assert isinstance(pipeline, Pipeline)
        assert pipeline._steps[-1]["type"] == "write"


class TestPipelineToConfig:
    """Tests for to_config."""

    def test_to_config_returns_list(self):
        pipeline = Pipeline().chunk_with("recursive", chunk_size=512)
        config = pipeline.to_config()
        assert isinstance(config, list)
        assert len(config) == 1
        assert config[0]["type"] == "chunk"
        assert config[0]["component"] == "recursive"
        assert config[0]["chunk_size"] == 512

    def test_to_config_saves_to_file(self, tmp_path):
        pipeline = Pipeline().process_with("text").chunk_with("recursive", chunk_size=256)
        out_path = tmp_path / "config.json"
        config = pipeline.to_config(out_path)
        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert len(loaded) == 2
        assert loaded == config

    def test_to_config_multiple_steps(self):
        pipeline = (
            Pipeline()
            .chunk_with("token", chunk_size=100)
            .refine_with("overlap", context_size=50)
            .export_with("json")
        )
        config = pipeline.to_config()
        assert len(config) == 3
        types = [c["type"] for c in config]
        assert types == ["chunk", "refine", "export"]


class TestPipelineDescribe:
    """Tests for describe and __repr__."""

    def test_describe_empty(self):
        assert Pipeline().describe() == "Empty pipeline"

    def test_describe_with_steps(self):
        pipeline = Pipeline().process_with("text").chunk_with("recursive")
        desc = pipeline.describe()
        # Describe uses CHOMP order; it also auto-inserts a default text chef
        # but since we added one, only one process step exists
        assert "process(text)" in desc
        assert "chunk(recursive)" in desc
        assert " -> " in desc

    def test_describe_auto_adds_chef(self):
        """Describe (via _reorder_steps) adds a default TextChef if none."""
        pipeline = Pipeline().chunk_with("recursive")
        desc = pipeline.describe()
        assert "process(text)" in desc
        assert "chunk(recursive)" in desc

    def test_repr(self):
        pipeline = Pipeline().chunk_with("recursive")
        r = repr(pipeline)
        assert r.startswith("Pipeline(")
        assert "chunk(recursive)" in r


class TestPipelineReset:
    """Tests for reset."""

    def test_reset_clears_steps(self):
        pipeline = Pipeline().chunk_with("recursive").process_with("text")
        assert len(pipeline._steps) == 2
        result = pipeline.reset()
        assert result is pipeline
        assert len(pipeline._steps) == 0
        assert len(pipeline._component_instances) == 0

    def test_reset_clears_cached_instances(self):
        pipeline = Pipeline().chunk_with("recursive", chunk_size=512)
        pipeline.run(texts="hello world")
        assert len(pipeline._component_instances) > 0
        pipeline.reset()
        assert len(pipeline._component_instances) == 0


class TestPipelineRunSkipsFetcher:
    """Test that run() skips fetch step when texts is provided."""

    def test_text_input_skips_fetcher(self):
        doc = (
            Pipeline()
            .fetch_from("file", path="/nonexistent/file.txt")
            .chunk_with("recursive", chunk_size=512)
            .run(texts="Direct text input")
        )
        # Should succeed despite nonexistent file path — fetcher is skipped
        assert isinstance(doc, Document)
        assert len(doc.chunks) > 0


class TestPipelineCallComponentEdgeCases:
    """Test _call_component for export/write/unknown step types."""

    def test_export_step_with_single_doc(self, tmp_path):
        """Export step with a single Document."""
        pipeline = (
            Pipeline()
            .chunk_with("token", chunk_size=512)
            .export_with("json", file=str(tmp_path / "out.jsonl"))
        )
        doc = pipeline.run(texts="Export test text.")
        assert isinstance(doc, Document)
        assert (tmp_path / "out.jsonl").exists()

    def test_export_step_with_multiple_docs(self, tmp_path):
        """Export step with list of Documents."""
        pipeline = (
            Pipeline()
            .chunk_with("token", chunk_size=512)
            .export_with("json", file=str(tmp_path / "out.jsonl"))
        )
        docs = pipeline.run(texts=["Doc 1.", "Doc 2."])
        assert isinstance(docs, list)
        assert (tmp_path / "out.jsonl").exists()

    def test_unknown_step_type_raises(self):
        """_call_component with unknown step type raises ValueError."""
        pipeline = Pipeline()
        component = MagicMock()
        with pytest.raises(ValueError, match="Unknown step type"):
            pipeline._call_component(component, "bogus", None, {})


class TestPipelineNonSerializableKwargs:
    """Test that non-JSON-serializable init kwargs fallback to repr for cache key."""

    def test_non_serializable_kwargs_cache_key(self):
        """Non-serializable kwargs should use repr-based cache key."""
        pipeline = Pipeline()

        # Create a step with a non-serializable kwarg value
        non_serializable = object()
        mock_component_info = MagicMock()
        mock_component_info.name = "TestComponent"
        mock_component_class = MagicMock()
        mock_component_info.component_class = mock_component_class

        step = {
            "type": "chunk",
            "component": mock_component_info,
            "kwargs": {"weird_param": non_serializable},
        }

        # Patch _split_parameters to return the non-serializable value as an init kwarg
        with patch.object(
            Pipeline,
            "_split_parameters",
            return_value=({"weird_param": non_serializable}, {}),
        ):
            instance, call_kwargs = pipeline._prepare_step_execution(step)
            # Should succeed — used repr fallback for cache key
            assert instance is mock_component_class.return_value


class TestPipelineAsync:
    """Tests for arun and _acall_component."""

    @pytest.mark.asyncio
    async def test_arun_no_steps(self):
        """Arun with no steps raises ValueError."""
        with pytest.raises(ValueError, match="no steps"):
            await Pipeline().arun(texts="test")

    @pytest.mark.asyncio
    async def test_arun_empty_list(self):
        """Arun with empty list returns empty list."""
        result = await Pipeline().chunk_with("recursive").arun(texts=[])
        assert result == []

    @pytest.mark.asyncio
    async def test_arun_single_text(self):
        """Arun with single text returns Document."""
        pipeline = Pipeline().chunk_with("token", chunk_size=512)
        doc = await pipeline.arun(texts="Async test text.")
        assert isinstance(doc, Document)
        assert len(doc.chunks) > 0

    @pytest.mark.asyncio
    async def test_arun_multiple_texts(self):
        """Arun with multiple texts returns list[Document]."""
        pipeline = Pipeline().chunk_with("token", chunk_size=512)
        docs = await pipeline.arun(texts=["Text A.", "Text B."])
        assert isinstance(docs, list)
        assert len(docs) == 2

    @pytest.mark.asyncio
    async def test_arun_skips_fetcher_with_text(self):
        """Arun skips fetcher when texts is provided."""
        pipeline = (
            Pipeline()
            .fetch_from("file", path="/nonexistent/file.txt")
            .chunk_with("token", chunk_size=512)
        )
        doc = await pipeline.arun(texts="Async text with skipped fetcher.")
        assert isinstance(doc, Document)

    @pytest.mark.asyncio
    async def test_arun_step_failure(self):
        """Arun wraps step exceptions in RuntimeError."""
        pipeline = Pipeline().chunk_with("recursive", chunk_size=512)
        # Pass an invalid type as input to force a failure
        with patch.object(Pipeline, "_reorder_steps") as mock_reorder:
            mock_reorder.return_value = [
                {
                    "type": "chunk",
                    "component": MagicMock(
                        name="MockChunker",
                        component_class=MagicMock,
                    ),
                    "kwargs": {},
                }
            ]
            with patch.object(Pipeline, "_validate_pipeline"):
                with patch.object(Pipeline, "_aexecute_step", side_effect=Exception("boom")):
                    with pytest.raises(RuntimeError, match="Pipeline failed at step"):
                        await pipeline.arun(texts="test")

    @pytest.mark.asyncio
    async def test_acall_component_unknown_step_type(self):
        """_acall_component with unknown step type raises ValueError."""
        pipeline = Pipeline()
        component = MagicMock()
        with pytest.raises(ValueError, match="Unknown step type"):
            await pipeline._acall_component(component, "bogus", None, {})

    @pytest.mark.asyncio
    async def test_acall_component_fetch(self):
        """_acall_component for fetch step."""
        pipeline = Pipeline()
        component = MagicMock()
        component.afetch = AsyncMock(return_value="fetched_data")
        result = await pipeline._acall_component(component, "fetch", None, {"key": "val"})
        component.afetch.assert_called_once_with(key="val")
        assert result == "fetched_data"

    @pytest.mark.asyncio
    async def test_acall_component_process_string(self):
        """_acall_component for process step with string input."""
        pipeline = Pipeline()
        component = MagicMock()
        component.aparse = AsyncMock(return_value="parsed")
        result = await pipeline._acall_component(component, "process", "hello", {})
        component.aparse.assert_called_once_with("hello")
        assert result == "parsed"

    @pytest.mark.asyncio
    async def test_acall_component_process_path(self):
        """_acall_component for process step with Path input."""
        pipeline = Pipeline()
        component = MagicMock()
        component.aprocess = AsyncMock(return_value="processed")
        result = await pipeline._acall_component(component, "process", Path("/tmp/f.txt"), {})
        component.aprocess.assert_called_once_with(Path("/tmp/f.txt"))
        assert result == "processed"

    @pytest.mark.asyncio
    async def test_acall_component_process_list(self):
        """_acall_component for process step with list input."""
        pipeline = Pipeline()
        component = MagicMock()
        component.aparse = AsyncMock(side_effect=lambda x: f"parsed_{x}")
        component.aprocess = AsyncMock(side_effect=lambda x: f"processed_{x}")
        result = await pipeline._acall_component(
            component, "process", ["text1", Path("/tmp/f.txt")], {}
        )
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_acall_component_chunk_single(self):
        """_acall_component for chunk step with single doc."""
        pipeline = Pipeline()
        component = MagicMock()
        doc = MagicMock()
        component.achunk_document = AsyncMock(return_value=doc)
        result = await pipeline._acall_component(component, "chunk", doc, {})
        assert result is doc

    @pytest.mark.asyncio
    async def test_acall_component_chunk_list(self):
        """_acall_component for chunk step with list of docs."""
        pipeline = Pipeline()
        component = MagicMock()
        docs = [MagicMock(), MagicMock()]
        component.achunk_document = AsyncMock(side_effect=lambda d: d)
        result = await pipeline._acall_component(component, "chunk", docs, {})
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_acall_component_refine_single(self):
        """_acall_component for refine step with single doc."""
        pipeline = Pipeline()
        component = MagicMock()
        doc = MagicMock()
        component.arefine_document = AsyncMock(return_value=doc)
        result = await pipeline._acall_component(component, "refine", doc, {})
        assert result is doc

    @pytest.mark.asyncio
    async def test_acall_component_refine_list(self):
        """_acall_component for refine step with list of docs."""
        pipeline = Pipeline()
        component = MagicMock()
        docs = [MagicMock(), MagicMock()]
        component.arefine_document = AsyncMock(side_effect=lambda d: d)
        result = await pipeline._acall_component(component, "refine", docs, {})
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_acall_component_export_single(self):
        """_acall_component for export step with single doc."""
        pipeline = Pipeline()
        component = MagicMock()
        doc = MagicMock()
        doc.chunks = [MagicMock()]
        component.aexport = AsyncMock()
        result = await pipeline._acall_component(component, "export", doc, {"file": "out.json"})
        component.aexport.assert_called_once()
        assert result is doc

    @pytest.mark.asyncio
    async def test_acall_component_export_list(self):
        """_acall_component for export step with list of docs."""
        pipeline = Pipeline()
        component = MagicMock()
        doc1 = MagicMock()
        doc1.chunks = [MagicMock()]
        doc2 = MagicMock()
        doc2.chunks = [MagicMock()]
        component.aexport = AsyncMock()
        result = await pipeline._acall_component(
            component, "export", [doc1, doc2], {"file": "out.json"}
        )
        assert result == [doc1, doc2]

    @pytest.mark.asyncio
    async def test_acall_component_write_single(self):
        """_acall_component for write step with single doc."""
        pipeline = Pipeline()
        component = MagicMock()
        doc = MagicMock()
        doc.chunks = [MagicMock()]
        component.awrite = AsyncMock(return_value="written")
        result = await pipeline._acall_component(component, "write", doc, {})
        component.awrite.assert_called_once()
        assert result == "written"

    @pytest.mark.asyncio
    async def test_acall_component_write_list(self):
        """_acall_component for write step with list of docs."""
        pipeline = Pipeline()
        component = MagicMock()
        doc1 = MagicMock()
        doc1.chunks = [MagicMock()]
        doc2 = MagicMock()
        doc2.chunks = [MagicMock()]
        component.awrite = AsyncMock(return_value="written")
        result = await pipeline._acall_component(component, "write", [doc1, doc2], {})
        assert result == "written"


class TestPipelineCallComponentWriteExport:
    """Test _call_component for write and export with list input."""

    def test_write_step_single_doc(self):
        pipeline = Pipeline()
        component = MagicMock()
        doc = MagicMock()
        doc.chunks = [MagicMock()]
        component.write.return_value = "result"
        result = pipeline._call_component(component, "write", doc, {})
        component.write.assert_called_once()
        assert result == "result"

    def test_write_step_list_of_docs(self):
        pipeline = Pipeline()
        component = MagicMock()
        doc1 = MagicMock()
        doc1.chunks = [MagicMock()]
        doc2 = MagicMock()
        doc2.chunks = [MagicMock()]
        component.write.return_value = "result"
        result = pipeline._call_component(component, "write", [doc1, doc2], {})
        assert result == "result"

    def test_export_step_list_of_docs(self):
        pipeline = Pipeline()
        component = MagicMock()
        doc1 = MagicMock()
        doc1.chunks = [MagicMock()]
        doc2 = MagicMock()
        doc2.chunks = [MagicMock()]
        result = pipeline._call_component(component, "export", [doc1, doc2], {"file": "x.json"})
        component.export.assert_called_once()
        assert result == [doc1, doc2]


class TestPipelineGetPositionalParams:
    """Test _get_positional_params static method."""

    def test_known_types(self):
        assert Pipeline._get_positional_params("fetch") == set()
        assert "text" in Pipeline._get_positional_params("process")
        assert "document" in Pipeline._get_positional_params("chunk")
        assert "chunks" in Pipeline._get_positional_params("refine")
        assert "chunks" in Pipeline._get_positional_params("export")
        assert "chunks" in Pipeline._get_positional_params("write")

    def test_unknown_type_returns_empty(self):
        assert Pipeline._get_positional_params("nonexistent") == set()


class TestPipelinePrepareStepRecipe:
    """Test _prepare_step_execution with recipe parameters."""

    def test_recipe_param_uses_from_recipe(self):
        """When kwargs has 'recipe', it calls from_recipe on the component class."""
        pipeline = Pipeline()

        class FakeComponent:
            @classmethod
            def from_recipe(cls, name, lang, **kwargs):
                return cls()

        mock_component_info = MagicMock()
        mock_component_info.name = "TestComp"
        mock_component_info.component_class = FakeComponent

        step = {
            "type": "chunk",
            "component": mock_component_info,
            "kwargs": {"recipe": "my_recipe", "lang": "fr"},
        }

        with patch.object(Pipeline, "_split_parameters", return_value=({}, {})):
            with patch.object(
                FakeComponent, "from_recipe", return_value=FakeComponent()
            ) as mock_fr:
                instance, call_kwargs = pipeline._prepare_step_execution(step)
                mock_fr.assert_called_once_with(name="my_recipe", lang="fr")
