"""Tests for code configuration types (MergeRule, SplitRule, LanguageConfig)."""

from chonkie.types.code import LanguageConfig, MergeRule, SplitRule

# ---------------------------------------------------------------------------
# MergeRule tests
# ---------------------------------------------------------------------------


def test_merge_rule_basic():
    """MergeRule stores name and node_types."""
    rule = MergeRule(name="imports", node_types=["import_statement", "import_from_statement"])
    assert rule.name == "imports"
    assert rule.node_types == ["import_statement", "import_from_statement"]


def test_merge_rule_default_bidirectional():
    """MergeRule defaults to bidirectional=False."""
    rule = MergeRule(name="r", node_types=["x"])
    assert rule.bidirectional is False


def test_merge_rule_bidirectional_true():
    """MergeRule can be created with bidirectional=True."""
    rule = MergeRule(name="r", node_types=["x"], bidirectional=True)
    assert rule.bidirectional is True


def test_merge_rule_default_text_pattern_is_none():
    """MergeRule defaults to text_pattern=None."""
    rule = MergeRule(name="r", node_types=["x"])
    assert rule.text_pattern is None


def test_merge_rule_with_text_pattern():
    """MergeRule accepts an optional regex text_pattern."""
    rule = MergeRule(name="r", node_types=["x"], text_pattern=r"^import\s")
    assert rule.text_pattern == r"^import\s"


def test_merge_rule_empty_node_types():
    """MergeRule can have an empty node_types list."""
    rule = MergeRule(name="empty", node_types=[])
    assert rule.node_types == []


def test_merge_rule_multiple_node_types():
    """MergeRule stores all provided node type strings."""
    types = ["a", "b", "c", "d"]
    rule = MergeRule(name="multi", node_types=types)
    assert rule.node_types == types


# ---------------------------------------------------------------------------
# SplitRule tests
# ---------------------------------------------------------------------------


def test_split_rule_string_body_child():
    """SplitRule stores a string body_child path."""
    rule = SplitRule(name="class_body", node_type="class_definition", body_child="class_body")
    assert rule.body_child == "class_body"


def test_split_rule_list_body_child():
    """SplitRule stores a list body_child path for nested traversal."""
    rule = SplitRule(
        name="nested",
        node_type="class_declaration",
        body_child=["class_declaration", "class_body"],
    )
    assert isinstance(rule.body_child, list)
    assert rule.body_child == ["class_declaration", "class_body"]


def test_split_rule_default_exclude_nodes_is_none():
    """SplitRule defaults to exclude_nodes=None."""
    rule = SplitRule(name="r", node_type="function_definition", body_child="block")
    assert rule.exclude_nodes is None


def test_split_rule_with_exclude_nodes():
    """SplitRule stores an optional list of excluded node types."""
    rule = SplitRule(
        name="r",
        node_type="function_definition",
        body_child="block",
        exclude_nodes=["comment", "decorator"],
    )
    assert rule.exclude_nodes == ["comment", "decorator"]


def test_split_rule_default_recursive_is_false():
    """SplitRule defaults to recursive=False."""
    rule = SplitRule(name="r", node_type="x", body_child="y")
    assert rule.recursive is False


def test_split_rule_recursive_true():
    """SplitRule can be created with recursive=True."""
    rule = SplitRule(name="r", node_type="x", body_child="y", recursive=True)
    assert rule.recursive is True


def test_split_rule_stores_node_type():
    """SplitRule stores the node_type field."""
    rule = SplitRule(name="func_split", node_type="function_definition", body_child="block")
    assert rule.node_type == "function_definition"


# ---------------------------------------------------------------------------
# LanguageConfig tests
# ---------------------------------------------------------------------------


def test_language_config_stores_language():
    """LanguageConfig stores the language string."""
    config = LanguageConfig(language="python", merge_rules=[], split_rules=[])
    assert config.language == "python"


def test_language_config_empty_rules():
    """LanguageConfig can be created with empty rule lists."""
    config = LanguageConfig(language="go", merge_rules=[], split_rules=[])
    assert config.merge_rules == []
    assert config.split_rules == []


def test_language_config_with_merge_rules():
    """LanguageConfig stores provided merge rules."""
    rules = [MergeRule(name="imports", node_types=["import_statement"])]
    config = LanguageConfig(language="python", merge_rules=rules, split_rules=[])
    assert len(config.merge_rules) == 1
    assert config.merge_rules[0].name == "imports"


def test_language_config_with_split_rules():
    """LanguageConfig stores provided split rules."""
    rules = [SplitRule(name="class", node_type="class_definition", body_child="class_body")]
    config = LanguageConfig(language="python", merge_rules=[], split_rules=rules)
    assert len(config.split_rules) == 1
    assert config.split_rules[0].name == "class"


def test_language_config_multiple_rules():
    """LanguageConfig stores multiple rules of each type."""
    merge_rules = [
        MergeRule(name="imports", node_types=["import_statement"]),
        MergeRule(name="decorators", node_types=["decorator"]),
    ]
    split_rules = [
        SplitRule(name="class", node_type="class_definition", body_child="class_body"),
        SplitRule(name="func", node_type="function_definition", body_child="block"),
    ]
    config = LanguageConfig(language="python", merge_rules=merge_rules, split_rules=split_rules)
    assert len(config.merge_rules) == 2
    assert len(config.split_rules) == 2


def test_language_config_equality():
    """Two LanguageConfigs with the same data are equal (dataclass default)."""
    cfg1 = LanguageConfig(language="go", merge_rules=[], split_rules=[])
    cfg2 = LanguageConfig(language="go", merge_rules=[], split_rules=[])
    assert cfg1 == cfg2


def test_language_config_inequality():
    """Two LanguageConfigs with different languages are not equal."""
    cfg1 = LanguageConfig(language="python", merge_rules=[], split_rules=[])
    cfg2 = LanguageConfig(language="rust", merge_rules=[], split_rules=[])
    assert cfg1 != cfg2
