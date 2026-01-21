"""Module containing CodeRegistry class.

This module provides a CodeRegistry class for managing code chunks.
"""

from collections.abc import KeysView

from chonkie.types.code import LanguageConfig, MergeRule, SplitRule


class _CodeLanguageRegistry:
    """Registry to store language configurations."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self.language_configs: dict[str, LanguageConfig] = {}

    def register(self, language: str, config: LanguageConfig) -> None:
        """Register a language configuration."""
        self.language_configs[language] = config

    def get(self, language: str) -> LanguageConfig:
        """Get a language configuration."""
        return self.language_configs[language]

    def __contains__(self, language: str) -> bool:
        """Check if a language is registered."""
        return language in self.language_configs

    def __getitem__(self, language: str) -> LanguageConfig:
        """Get a language configuration."""
        return self.language_configs[language]

    def keys(self) -> KeysView[str]:
        """Get all registered language keys."""
        return self.language_configs.keys()


# Initialize the registry
CodeLanguageRegistry = _CodeLanguageRegistry()

# Register the python language config
pyconfig = LanguageConfig(
    language="python",
    merge_rules=[
        MergeRule(
            name="import_group",
            node_types=["import_from_statement", "import_statement"],
            text_pattern=None,
            bidirectional=True,
        ),
        MergeRule(
            name="comment_group",
            node_types=["comment"],
            text_pattern=None,
            bidirectional=True,
        ),
        MergeRule(
            name="comment_class_group",
            node_types=["comment", "class_definition"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_function_group",
            node_types=["comment", "function_definition"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="docstring_import_group",
            node_types=["expression_statement", "import_statement"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="docstring_import_from_group",
            node_types=["expression_statement", "import_from_statement"],
            text_pattern=None,
            bidirectional=False,
        ),
    ],
    split_rules=[
        SplitRule(name="class_definition", node_type="class_definition", body_child="block"),
    ],
)
CodeLanguageRegistry.register("python", pyconfig)

# Adding the typescript language config
tsconfig = LanguageConfig(
    language="typescript",
    merge_rules=[
        # Group comments with following constructs
        MergeRule(
            name="comment_export_group",
            node_types=["comment", "export_statement"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_interface_group",
            node_types=["comment", "interface_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_type_alias_group",
            node_types=["comment", "type_alias_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_class_group",
            node_types=["comment", "class_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_function_group",
            node_types=["comment", "function_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_method_group",
            node_types=["comment", "method_definition"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Group similar constructs
        MergeRule(
            name="import_group",
            node_types=["import_statement"],
            text_pattern=None,
            bidirectional=True,
        ),
        MergeRule(
            name="type_definitions_group",
            node_types=["interface_declaration", "type_alias_declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group lexical declarations (const, let, var)
        MergeRule(
            name="lexical_declaration_group",
            node_types=["lexical_declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group expression statements
        MergeRule(
            name="expression_statement_group",
            node_types=["expression_statement"],
            text_pattern=None,
            bidirectional=True,
        ),
    ],
    split_rules=[
        # Split large standalone class declarations
        SplitRule(name="class_split", node_type="class_declaration", body_child="class_body"),
    ],
)
CodeLanguageRegistry.register("typescript", tsconfig)

# Adding the JavaScript language config based on tree-sitter analysis
jsconfig = LanguageConfig(
    language="javascript",
    merge_rules=[
        # Group comments with following constructs (based on actual AST analysis)
        MergeRule(
            name="comment_class_group",
            node_types=["comment", "class_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_function_group",
            node_types=["comment", "function_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_lexical_group",
            node_types=["comment", "lexical_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_variable_group",
            node_types=["comment", "variable_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_export_group",
            node_types=["comment", "export_statement"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_import_group",
            node_types=["comment", "import_statement"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_expression_group",
            node_types=["comment", "expression_statement"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Group similar statement types
        MergeRule(
            name="import_statement_group",
            node_types=["import_statement"],
            text_pattern=None,
            bidirectional=True,
        ),
        MergeRule(
            name="lexical_declaration_group",
            node_types=["lexical_declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
        MergeRule(
            name="variable_declaration_group",
            node_types=["variable_declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
        MergeRule(
            name="export_statement_group",
            node_types=["export_statement"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group require expressions (CommonJS pattern)
        MergeRule(
            name="require_group",
            node_types=["lexical_declaration"],
            text_pattern=r".*require\s*\(",
            bidirectional=True,
        ),
        # Group module.exports expressions
        MergeRule(
            name="module_exports_group",
            node_types=["expression_statement"],
            text_pattern=r".*module\.exports.*",
            bidirectional=True,
        ),
    ],
    split_rules=[
        # Split large class declarations at class_body, excluding structural punctuation
        SplitRule(
            name="class_split",
            node_type="class_declaration",
            body_child="class_body",
            exclude_nodes=["{", "}", ";", ","],
        ),
    ],
)
CodeLanguageRegistry.register("javascript", jsconfig)

# Adding the Rust language config based on tree-sitter analysis
rustconfig = LanguageConfig(
    language="rust",
    merge_rules=[
        # Comment merging patterns (based on AST analysis showing most frequent patterns)
        # Comments with attributes (10 occurrences) - this is the most frequent pattern in Rust
        MergeRule(
            name="comment_attribute_group",
            node_types=["line_comment", "attribute_item"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="block_comment_attribute_group",
            node_types=["block_comment", "attribute_item"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Attributes with type definitions and functions (this will chain with comment-attribute merges)
        MergeRule(
            name="attribute_struct_group",
            node_types=["attribute_item", "struct_item"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="attribute_enum_group",
            node_types=["attribute_item", "enum_item"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="attribute_trait_group",
            node_types=["attribute_item", "trait_item"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="attribute_function_group",
            node_types=["attribute_item", "function_item"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="attribute_mod_group",
            node_types=["attribute_item", "mod_item"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Merge attributes with expression statements
        MergeRule(
            name="attribute_expression_group",
            node_types=["attribute_item", "expression_statement"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Direct comment merging for cases without attributes
        MergeRule(
            name="comment_struct_group",
            node_types=["line_comment", "struct_item"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="block_comment_struct_group",
            node_types=["block_comment", "struct_item"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_enum_group",
            node_types=["line_comment", "enum_item"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="block_comment_enum_group",
            node_types=["block_comment", "enum_item"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_trait_group",
            node_types=["line_comment", "trait_item"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="block_comment_trait_group",
            node_types=["block_comment", "trait_item"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Comments with functions/methods (8 occurrences)
        MergeRule(
            name="comment_function_group",
            node_types=["line_comment", "function_item"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="block_comment_function_group",
            node_types=["block_comment", "function_item"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Comments with modules (1 occurrence)
        MergeRule(
            name="comment_module_group",
            node_types=["line_comment", "mod_item"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="block_comment_module_group",
            node_types=["block_comment", "mod_item"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Comments with macro definitions (1 occurrence)
        MergeRule(
            name="comment_macro_group",
            node_types=["line_comment", "macro_definition"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="block_comment_macro_group",
            node_types=["block_comment", "macro_definition"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Comments with use declarations (3 occurrences)
        MergeRule(
            name="comment_use_group",
            node_types=["line_comment", "use_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="block_comment_use_group",
            node_types=["block_comment", "use_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Statement grouping patterns
        # Group use declarations (imports) - found 10 use declarations
        MergeRule(
            name="use_declaration_group",
            node_types=["use_declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group type definitions (structs, enums, traits)
        MergeRule(
            name="type_definition_group",
            node_types=["struct_item", "enum_item", "trait_item", "type_item"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group attribute items (derives, cfg, etc.)
        MergeRule(
            name="attribute_group",
            node_types=["attribute_item"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group macro invocations
        MergeRule(
            name="macro_invocation_group",
            node_types=["macro_invocation"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group impl blocks for same type
        MergeRule(
            name="impl_block_group",
            node_types=["impl_item"],
            text_pattern=None,
            bidirectional=True,
        ),
    ],
    split_rules=[
        # Split large constructs at their body containers (based on AST analysis)
        # Split struct definitions at field_declaration_list (found 8 structs)
        SplitRule(
            name="struct_split",
            node_type="struct_item",
            body_child="field_declaration_list",
            exclude_nodes=["{", "}", ";", ","],
        ),
        # Split impl blocks at declaration_list (found 5 impl blocks)
        SplitRule(
            name="impl_split",
            node_type="impl_item",
            body_child="declaration_list",
            exclude_nodes=["{", "}", ";", ","],
        ),
        # Split enum definitions at enum_variant_list (found 3 enums)
        SplitRule(
            name="enum_split",
            node_type="enum_item",
            body_child="enum_variant_list",
            exclude_nodes=["{", "}", ";", ","],
        ),
        # Split trait definitions at declaration_list (found 1 trait)
        SplitRule(
            name="trait_split",
            node_type="trait_item",
            body_child="declaration_list",
            exclude_nodes=["{", "}", ";", ","],
        ),
        # Functions should never be split - keep them as coherent units
        # SplitRule for function_item removed to preserve semantic integrity
        # Split module definitions at declaration_list (found 2 modules)
        SplitRule(
            name="module_split",
            node_type="mod_item",
            body_child="declaration_list",
            exclude_nodes=["{", "}", ";", ","],
        ),
    ],
)
CodeLanguageRegistry.register("rust", rustconfig)

# Adding the Go language config based on tree-sitter analysis
goconfig = LanguageConfig(
    language="go",
    merge_rules=[
        # Comment merging patterns (based on AST analysis frequency)
        # Comments with method declarations (23 occurrences) - most frequent pattern
        MergeRule(
            name="comment_method_group",
            node_types=["comment", "method_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Comments with type declarations (10 occurrences) - structs, interfaces, etc.
        MergeRule(
            name="comment_type_group",
            node_types=["comment", "type_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Comments with function declarations (9 occurrences) - standalone functions
        MergeRule(
            name="comment_function_group",
            node_types=["comment", "function_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Note: Removed merge rules for split function/method pieces
        # since functions and methods are no longer split
        # Comments with package clause (1 occurrence) - package documentation
        MergeRule(
            name="comment_package_group",
            node_types=["comment", "package_clause"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Merge comments with statement types
        MergeRule(
            name="comment_return_group",
            node_types=["comment", "return_statement"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_expression_group",
            node_types=["comment", "expression_statement"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_assignment_group",
            node_types=["comment", "assignment_statement"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_if_group",
            node_types=["comment", "if_statement"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_for_group",
            node_types=["comment", "for_statement"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_var_decl_group",
            node_types=["comment", "short_var_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Statement grouping patterns
        # Group import declarations and specs (found 1 import_declaration with 7 import_specs)
        MergeRule(
            name="import_declaration_group",
            node_types=["import_declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
        MergeRule(
            name="import_spec_group",
            node_types=["import_spec"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group type declarations (found 9 type declarations)
        MergeRule(
            name="type_declaration_group",
            node_types=["type_declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group methods by receiver type (method_declaration grouped by receiver)
        # This requires text pattern matching for methods with same receiver
        MergeRule(
            name="receiver_method_group",
            node_types=["method_declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group constructor functions (New* functions)
        MergeRule(
            name="constructor_function_group",
            node_types=["function_declaration"],
            text_pattern=r"^func New.*",
            bidirectional=True,
        ),
        # Group related statements within functions/methods
        MergeRule(
            name="var_declaration_group",
            node_types=["short_var_declaration", "var_declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group const declarations
        MergeRule(
            name="const_declaration_group",
            node_types=["const_declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
    ],
    split_rules=[
        # Split large constructs at their body containers (based on AST analysis)
        # Split struct types at field_declaration_list (found 9 struct types)
        SplitRule(
            name="struct_split",
            node_type="struct_type",
            body_child="field_declaration_list",
            exclude_nodes=["{", "}", ",", ";"],
        ),
        # Split interface types at method_spec_list (found 1 interface type)
        SplitRule(
            name="interface_split",
            node_type="interface_type",
            body_child="method_spec_list",
            exclude_nodes=["{", "}", ",", ";"],
        ),
        # Functions and methods should never be split - keep them as coherent units
        # SplitRule for function_declaration removed to preserve semantic integrity
        # SplitRule for method_declaration removed to preserve semantic integrity
    ],
)
CodeLanguageRegistry.register("go", goconfig)

# Adding the Java language config based on tree-sitter analysis
javaconfig = LanguageConfig(
    language="java",
    merge_rules=[
        # Comment merging patterns (based on AST analysis frequency)
        # Comments with method declarations (23 occurrences) - most frequent pattern
        MergeRule(
            name="comment_method_group",
            node_types=["block_comment", "method_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="line_comment_method_group",
            node_types=["line_comment", "method_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Comments with field declarations (13 occurrences) - second most frequent
        MergeRule(
            name="comment_field_group",
            node_types=["block_comment", "field_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Comments with class declarations (5 occurrences)
        MergeRule(
            name="comment_class_group",
            node_types=["block_comment", "class_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Comments with constructor declarations (3 occurrences)
        MergeRule(
            name="comment_constructor_group",
            node_types=["block_comment", "constructor_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="line_comment_constructor_group",
            node_types=["line_comment", "constructor_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Comments with interface and enum declarations
        MergeRule(
            name="comment_interface_group",
            node_types=["block_comment", "interface_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_enum_group",
            node_types=["block_comment", "enum_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Comments with package declaration
        MergeRule(
            name="comment_package_group",
            node_types=["block_comment", "package_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Merge comments with statement types
        MergeRule(
            name="comment_local_var_group",
            node_types=["line_comment", "local_variable_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_expression_group",
            node_types=["line_comment", "expression_statement"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_if_group",
            node_types=["line_comment", "if_statement"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_return_group",
            node_types=["line_comment", "return_statement"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Annotation merging patterns (Java-specific)
        # Merge modifiers (which contain annotations) with declarations
        MergeRule(
            name="modifiers_class_group",
            node_types=["modifiers", "class_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="modifiers_method_group",
            node_types=["modifiers", "method_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="modifiers_field_group",
            node_types=["modifiers", "field_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Statement grouping patterns
        # Group import declarations (found 14 import declarations)
        MergeRule(
            name="import_declaration_group",
            node_types=["import_declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group type declarations
        MergeRule(
            name="type_declaration_group",
            node_types=["class_declaration", "interface_declaration", "enum_declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group getter/setter methods (Java pattern)
        MergeRule(
            name="getter_method_group",
            node_types=["method_declaration"],
            text_pattern=r"public\s+\w+\s+get\w+\(\)",
            bidirectional=True,
        ),
        MergeRule(
            name="setter_method_group",
            node_types=["method_declaration"],
            text_pattern=r"public\s+void\s+set\w+\(",
            bidirectional=True,
        ),
        # Group constructors
        MergeRule(
            name="constructor_group",
            node_types=["constructor_declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group field declarations by access level
        MergeRule(
            name="private_field_group",
            node_types=["field_declaration"],
            text_pattern=r"private\s+",
            bidirectional=True,
        ),
        MergeRule(
            name="public_field_group",
            node_types=["field_declaration"],
            text_pattern=r"public\s+",
            bidirectional=True,
        ),
    ],
    split_rules=[
        # Split large constructs at their body containers (based on AST analysis)
        # Split class declarations at class_body (found 5 class declarations)
        SplitRule(
            name="class_split",
            node_type="class_declaration",
            body_child="class_body",
            exclude_nodes=["{", "}", ";", ","],
        ),
        # Split interface declarations at interface_body (found 1 interface declaration)
        SplitRule(
            name="interface_split",
            node_type="interface_declaration",
            body_child="interface_body",
            exclude_nodes=["{", "}", ";", ","],
        ),
        # Split enum declarations at enum_body (found 1 enum declaration)
        SplitRule(
            name="enum_split",
            node_type="enum_declaration",
            body_child="enum_body",
            exclude_nodes=["{", "}", ";", ","],
        ),
        # Methods and constructors should never be split - keep them as coherent units
        # SplitRule for method_declaration removed to preserve semantic integrity
        # SplitRule for constructor_declaration removed to preserve semantic integrity
    ],
)
CodeLanguageRegistry.register("java", javaconfig)

# Adding the Markdown language config based on tree-sitter analysis
markdownconfig = LanguageConfig(
    language="markdown",
    merge_rules=[
        # No merge rules to test pure splitting
    ],
    split_rules=[
        # Recursively split sections at nested sections to break up large documents
        # This preserves semantic units (tables, code blocks) while allowing section splitting
        SplitRule(
            name="section_recursive_split",
            node_type="section",
            body_child="section",
            exclude_nodes=[],
            recursive=True,
        ),
    ],
)
CodeLanguageRegistry.register("markdown", markdownconfig)

# Adding the HTML language config
htmlconfig = LanguageConfig(
    language="html",
    merge_rules=[
        # Group HTML comments with following elements
        MergeRule(
            name="comment_element_group",
            node_types=["comment", "element"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Group meta tags in head (only small meta tags)
        MergeRule(
            name="meta_tag_group",
            node_types=["element"],
            text_pattern=r".*<meta.*",
            bidirectional=True,
        ),
        # Group link tags (CSS, favicons, etc.)
        MergeRule(
            name="link_tag_group",
            node_types=["element"],
            text_pattern=r".*<link.*",
            bidirectional=True,
        ),
    ],
    split_rules=[
        # Split large HTML elements at their child elements (recursively)
        SplitRule(
            name="html_element_split",
            node_type="element",
            body_child="element",
            exclude_nodes=["start_tag", "end_tag", "self_closing_tag"],
            recursive=True,
        ),
    ],
)
CodeLanguageRegistry.register("html", htmlconfig)

# Adding the CSS language config
cssconfig = LanguageConfig(
    language="css",
    merge_rules=[
        # Group CSS comments with following rules
        MergeRule(
            name="comment_rule_group",
            node_types=["comment", "rule_set"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_at_rule_group",
            node_types=["comment", "at_rule"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Group related selectors
        MergeRule(
            name="selector_group",
            node_types=["rule_set"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group @import statements
        MergeRule(
            name="import_group",
            node_types=["at_rule"],
            text_pattern=r"@import.*",
            bidirectional=True,
        ),
        # Group @media queries
        MergeRule(
            name="media_query_group",
            node_types=["at_rule"],
            text_pattern=r"@media.*",
            bidirectional=True,
        ),
        # Group @keyframes rules
        MergeRule(
            name="keyframes_group",
            node_types=["at_rule"],
            text_pattern=r"@keyframes.*",
            bidirectional=True,
        ),
        # Group CSS custom properties (variables)
        MergeRule(
            name="custom_property_group",
            node_types=["declaration"],
            text_pattern=r"--.*:",
            bidirectional=True,
        ),
    ],
    split_rules=[
        # Split large rule sets at declaration blocks
        SplitRule(
            name="rule_set_split",
            node_type="rule_set",
            body_child="declaration",
            exclude_nodes=["{", "}", ";"],
        ),
        # Split @media rules at their content
        SplitRule(
            name="media_rule_split",
            node_type="at_rule",
            body_child="rule_set",
            exclude_nodes=["{", "}", "@media"],
        ),
    ],
)
CodeLanguageRegistry.register("css", cssconfig)

# Adding the C language config
cconfig = LanguageConfig(
    language="c",
    merge_rules=[
        # Group comments with following declarations
        MergeRule(
            name="comment_function_group",
            node_types=["comment", "function_definition"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_declaration_group",
            node_types=["comment", "declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_struct_group",
            node_types=["comment", "struct_specifier"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Group preprocessor directives
        MergeRule(
            name="include_group",
            node_types=["preproc_include"],
            text_pattern=None,
            bidirectional=True,
        ),
        MergeRule(
            name="define_group",
            node_types=["preproc_def"],
            text_pattern=None,
            bidirectional=True,
        ),
        MergeRule(
            name="ifdef_group",
            node_types=["preproc_ifdef", "preproc_if"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group variable declarations
        MergeRule(
            name="variable_declaration_group",
            node_types=["declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group struct/union/enum definitions
        MergeRule(
            name="type_definition_group",
            node_types=["struct_specifier", "union_specifier", "enum_specifier"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group typedefs
        MergeRule(
            name="typedef_group",
            node_types=["type_definition"],
            text_pattern=None,
            bidirectional=True,
        ),
    ],
    split_rules=[
        # Split large structs at field declarations
        SplitRule(
            name="struct_split",
            node_type="struct_specifier",
            body_child="field_declaration",
            exclude_nodes=["{", "}", ";", "struct"],
        ),
        # Split large unions at field declarations
        SplitRule(
            name="union_split",
            node_type="union_specifier",
            body_child="field_declaration",
            exclude_nodes=["{", "}", ";", "union"],
        ),
        # Split large enums at enumerator declarations
        SplitRule(
            name="enum_split",
            node_type="enum_specifier",
            body_child="enumerator",
            exclude_nodes=["{", "}", ";", ",", "enum"],
        ),
        # Functions should not be split to preserve semantic integrity
    ],
)
CodeLanguageRegistry.register("c", cconfig)

# Adding the C++ language config
cppconfig = LanguageConfig(
    language="cpp",
    merge_rules=[
        # Group comments with following declarations (C++ specific)
        MergeRule(
            name="comment_function_group",
            node_types=["comment", "function_definition"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_class_group",
            node_types=["comment", "class_specifier"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_template_group",
            node_types=["comment", "template_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_namespace_group",
            node_types=["comment", "namespace_definition"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Group preprocessor directives (inherited from C)
        MergeRule(
            name="include_group",
            node_types=["preproc_include"],
            text_pattern=None,
            bidirectional=True,
        ),
        MergeRule(
            name="define_group",
            node_types=["preproc_def"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group using declarations and directives
        MergeRule(
            name="using_group",
            node_types=["using_declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
        MergeRule(
            name="namespace_using_group",
            node_types=["using_declaration"],
            text_pattern=r"using namespace.*",
            bidirectional=True,
        ),
        # Group template specializations
        MergeRule(
            name="template_group",
            node_types=["template_declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group access specifiers with following members
        MergeRule(
            name="access_member_group",
            node_types=["access_specifier", "field_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="access_function_group",
            node_types=["access_specifier", "function_definition"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Group constructor/destructor declarations
        MergeRule(
            name="constructor_group",
            node_types=["function_definition"],
            text_pattern=r".*::\w+\s*\(",
            bidirectional=True,
        ),
        MergeRule(
            name="destructor_group",
            node_types=["function_definition"],
            text_pattern=r".*::~\w+\s*\(",
            bidirectional=True,
        ),
    ],
    split_rules=[
        # Split large classes at member declarations
        SplitRule(
            name="class_split",
            node_type="class_specifier",
            body_child="field_declaration",
            exclude_nodes=["{", "}", ";", "class", "struct", "public:", "private:", "protected:"],
        ),
        # Split large structs (C++ style)
        SplitRule(
            name="struct_split",
            node_type="struct_specifier",
            body_child="field_declaration",
            exclude_nodes=["{", "}", ";", "struct"],
        ),
        # Split namespaces at their declarations
        SplitRule(
            name="namespace_split",
            node_type="namespace_definition",
            body_child="declaration",
            exclude_nodes=["{", "}", "namespace"],
        ),
        # Functions and methods should not be split to preserve semantic integrity
    ],
)
CodeLanguageRegistry.register("cpp", cppconfig)

# Adding the C# language config
csharpconfig = LanguageConfig(
    language="csharp",
    merge_rules=[
        # Group comments with following declarations
        MergeRule(
            name="comment_class_group",
            node_types=["comment", "class_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_method_group",
            node_types=["comment", "method_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_property_group",
            node_types=["comment", "property_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_field_group",
            node_types=["comment", "field_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_interface_group",
            node_types=["comment", "interface_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_enum_group",
            node_types=["comment", "enum_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="comment_namespace_group",
            node_types=["comment", "namespace_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Group using directives
        MergeRule(
            name="using_directive_group",
            node_types=["using_directive"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group attributes with declarations
        MergeRule(
            name="attribute_class_group",
            node_types=["attribute_list", "class_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="attribute_method_group",
            node_types=["attribute_list", "method_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        MergeRule(
            name="attribute_property_group",
            node_types=["attribute_list", "property_declaration"],
            text_pattern=None,
            bidirectional=False,
        ),
        # Group properties by type (get/set patterns)
        MergeRule(
            name="auto_property_group",
            node_types=["property_declaration"],
            text_pattern=r".*{\s*get;\s*set;\s*}",
            bidirectional=True,
        ),
        # Group constructor declarations
        MergeRule(
            name="constructor_group",
            node_types=["constructor_declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group event declarations
        MergeRule(
            name="event_declaration_group",
            node_types=["event_declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group delegate declarations
        MergeRule(
            name="delegate_declaration_group",
            node_types=["delegate_declaration"],
            text_pattern=None,
            bidirectional=True,
        ),
        # Group field declarations by access modifier
        MergeRule(
            name="private_field_group",
            node_types=["field_declaration"],
            text_pattern=r"private\s+",
            bidirectional=True,
        ),
        MergeRule(
            name="public_field_group",
            node_types=["field_declaration"],
            text_pattern=r"public\s+",
            bidirectional=True,
        ),
    ],
    split_rules=[
        # Split large classes at member declarations
        SplitRule(
            name="class_split",
            node_type="class_declaration",
            body_child="class_body",
            exclude_nodes=["{", "}", ";", "class"],
        ),
        # Split large interfaces at member declarations
        SplitRule(
            name="interface_split",
            node_type="interface_declaration",
            body_child="interface_body",
            exclude_nodes=["{", "}", ";", "interface"],
        ),
        # Split large enums at enum members
        SplitRule(
            name="enum_split",
            node_type="enum_declaration",
            body_child="enum_member_declaration",
            exclude_nodes=["{", "}", ";", ",", "enum"],
        ),
        # Split namespaces at their member declarations
        SplitRule(
            name="namespace_split",
            node_type="namespace_declaration",
            body_child="namespace_body",
            exclude_nodes=["{", "}", "namespace"],
        ),
        # Methods, properties, and constructors should not be split to preserve semantic integrity
    ],
)
CodeLanguageRegistry.register("csharp", csharpconfig)
