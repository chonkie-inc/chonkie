from __future__ import annotations

import ctypes
import sys
from importlib import import_module
from pathlib import Path
from typing import Literal, cast

import tree_sitter_c_sharp
import tree_sitter_embedded_template
import tree_sitter_yaml
from tree_sitter import Language, Parser

SupportedLanguage = Literal[
    "actionscript",
    "ada",
    "agda",
    "apex",
    "arduino",
    "asm",
    "astro",
    "bash",
    "beancount",
    "bibtex",
    "bicep",
    "bitbake",
    "bsl",
    "c",
    "cairo",
    "capnp",
    "chatito",
    "clarity",
    "clojure",
    "cmake",
    "cobol",
    "comment",
    "commonlisp",
    "cpon",
    "cpp",
    "csharp",
    "css",
    "csv",
    "cuda",
    "d",
    "dart",
    "dockerfile",
    "doxygen",
    "dtd",
    "elisp",
    "elixir",
    "elm",
    "embeddedtemplate",
    "erlang",
    "fennel",
    "firrtl",
    "fish",
    "fortran",
    "fsharp",
    "fsharp_signature",
    "func",
    "gdscript",
    "gitattributes",
    "gitcommit",
    "gitignore",
    "gleam",
    "glsl",
    "gn",
    "go",
    "gomod",
    "gosum",
    "graphql",
    "groovy",
    "gstlaunch",
    "hack",
    "hare",
    "haskell",
    "haxe",
    "hcl",
    "heex",
    "hlsl",
    "html",
    "hyprlang",
    "ini",
    "ispc",
    "janet",
    "java",
    "javascript",
    "jsdoc",
    "json",
    "jsonnet",
    "julia",
    "kconfig",
    "kdl",
    "kotlin",
    "latex",
    "linkerscript",
    "llvm",
    "lua",
    "luadoc",
    "luap",
    "luau",
    "make",
    "markdown",
    "markdown_inline",
    "matlab",
    "mermaid",
    "meson",
    "netlinx",
    "nim",
    "ninja",
    "nix",
    "nqc",
    "objc",
    "ocaml",
    "ocaml_interface",
    "odin",
    "org",
    "pascal",
    "pem",
    "perl",
    "pgn",
    "php",
    "po",
    "pony",
    "powershell",
    "printf",
    "prisma",
    "properties",
    "proto",
    "psv",
    "puppet",
    "purescript",
    "pymanifest",
    "python",
    "qmldir",
    "qmljs",
    "query",
    "r",
    "racket",
    "re2c",
    "readline",
    "rego",
    "requirements",
    "ron",
    "rst",
    "ruby",
    "rust",
    "scala",
    "scheme",
    "scss",
    "smali",
    "smithy",
    "solidity",
    "sparql",
    "swift",
    "sql",
    "squirrel",
    "starlark",
    "svelte",
    "tablegen",
    "tcl",
    "terraform",
    "test",
    "thrift",
    "toml",
    "tsv",
    "tsx",
    "twig",
    "typescript",
    "typst",
    "udev",
    "ungrammar",
    "uxntal",
    "v",
    "verilog",
    "vhdl",
    "vim",
    "vue",
    "wast",
    "wat",
    "wgsl",
    "xcompose",
    "xml",
    "yaml",
    "yuck",
    "zig",
    "magik",
]


def get_binding(language_name: SupportedLanguage) -> object:
    """Get the binding for the given language name.

    Args:
        language_name: The name of the language.

    Raises:
        LookupError: If the language is not found.

    Returns:
        A pycapsule object
    """
    if language_name == "yaml":
        return tree_sitter_yaml.language()

    if language_name == "csharp":
        return tree_sitter_c_sharp.language()

    if language_name == "embeddedtemplate":
        return tree_sitter_embedded_template.language()

    try:
        module = import_module(name=f".bindings.{language_name}", package=__package__)
        return cast("object", module.language())
    except (ModuleNotFoundError, ImportError) as e:
        # Workaround for Windows environments where wheels built with one Python version (e.g., 3.12)
        # and installed on a different version (e.g., 3.10) fail with "DLL load failed while importing".
        # This may be a Python bug, handling cases where the normal import_module mechanism fails.
        package_path = Path(__file__).parent
        ext = ".pyd" if sys.platform.startswith("win") else ".so"
        lib_path = package_path / "bindings" / f"{language_name}{ext}"

        if lib_path.exists():
            lib = ctypes.cdll.LoadLibrary(str(lib_path))
            language_fn = getattr(lib, f"tree_sitter_{language_name}", None)
            if language_fn:
                return language_fn()

        raise LookupError(f"Could not find language library for {language_name}") from e


def get_language(language_name: SupportedLanguage) -> Language:
    """Get the language with the given name.

    Args:
        language_name: The name of the language.

    Returns:
        Language: The language as a tree-sitter Language instance.
    """
    binding = get_binding(language_name)
    return Language(binding)


def get_parser(language_name: SupportedLanguage) -> Parser:
    """Get a parser for the given language name.

    Args:
        language_name: The name of the language.

    Returns:
        Parser: The parser for the language as a tree-sitter Parser instance.
    """
    return Parser(get_language(language_name=language_name))


__all__ = ["SupportedLanguage", "get_binding", "get_language", "get_parser"]
