"""Configuration system for CHOMP pipelines.

This module provides utilities to serialize, deserialize, save, and load CHOMP pipeline
configurations. It enables users to create pipeline configurations once and reuse them
or share them with others.
"""

import json
import os
import pickle
import inspect
import importlib
from typing import Any, Dict, Optional, Union, List, Type, cast

from ..chefs.base import BaseChef
from ..chunker import BaseChunker
from ..refinery import BaseRefinery
from ..friends import BasePorter, BaseHandshake
from ..types import RecursiveRules, RecursiveLevel


class ChompConfig:
    """Configuration for CHOMP pipelines.
    
    This class handles serialization and deserialization of CHOMP pipeline configurations,
    allowing users to save and load their pipeline setups.
    """
    
    # Types that need special handling during serialization/deserialization
    SPECIAL_TYPES = {
        "RecursiveRules": "src.chonkie.types.recursive",
        "RecursiveLevel": "src.chonkie.types.recursive",
        "Tokenizer": "src.chonkie.tokenizer"
    }
    
    @staticmethod
    def _serialize_param(param: Any) -> Any:
        """Serialize a parameter value.
        
        Handles special cases like objects that need custom serialization.
        
        Args:
            param: The parameter value to serialize
            
        Returns:
            A serialized version of the parameter value
        """
        # Handle None values
        if param is None:
            return None
            
        # Handle builtin types
        if isinstance(param, (str, int, float, bool)):
            return param
            
        # Handle lists
        if isinstance(param, list):
            return [ChompConfig._serialize_param(item) for item in param]
            
        # Handle dictionaries
        if isinstance(param, dict):
            return {k: ChompConfig._serialize_param(v) for k, v in param.items()}
            
        # Handle known special types
        param_type = type(param).__name__
        if param_type in ChompConfig.SPECIAL_TYPES:
            # For RecursiveRules and RecursiveLevel, we want to preserve their initialization parameters
            if param_type == "RecursiveRules":
                return {
                    "__type__": param_type,
                    "__module__": ChompConfig.SPECIAL_TYPES[param_type],
                    "levels": ChompConfig._serialize_param(param.levels)
                }
            elif param_type == "RecursiveLevel":
                return {
                    "__type__": param_type,
                    "__module__": ChompConfig.SPECIAL_TYPES[param_type],
                    "delimiters": param.delimiters,
                    "whitespace": param.whitespace,
                    "include_delim": param.include_delim
                }
            elif param_type == "Tokenizer":
                return {
                    "__type__": param_type,
                    "__module__": ChompConfig.SPECIAL_TYPES[param_type]
                }
                
        # Default fallback: convert to string
        return str(param)
    
    @staticmethod
    def _deserialize_param(param: Any) -> Any:
        """Deserialize a parameter value.
        
        Handles special cases like reconstructing complex objects.
        
        Args:
            param: The serialized parameter value
            
        Returns:
            Deserialized parameter value
        """
        # Handle None values
        if param is None:
            return None
            
        # Handle builtin types
        if isinstance(param, (str, int, float, bool)):
            return param
            
        # Handle lists
        if isinstance(param, list):
            return [ChompConfig._deserialize_param(item) for item in param]
            
        # Handle dictionaries that may contain special types
        if isinstance(param, dict):
            # Check if this is a special type dictionary
            if "__type__" in param and "__module__" in param:
                type_name = param["__type__"]
                module_name = param["__module__"]
                
                # Handle RecursiveRules
                if type_name == "RecursiveRules":
                    # Import the class
                    module = importlib.import_module(module_name)
                    cls = getattr(module, type_name)
                    levels = ChompConfig._deserialize_param(param.get("levels", []))
                    return cls(levels=levels)
                    
                # Handle RecursiveLevel
                elif type_name == "RecursiveLevel":
                    module = importlib.import_module(module_name)
                    cls = getattr(module, type_name)
                    return cls(
                        delimiters=param.get("delimiters"),
                        whitespace=param.get("whitespace", False),
                        include_delim=param.get("include_delim", "prev")
                    )
                    
                # Handle Tokenizer
                elif type_name == "Tokenizer":
                    module = importlib.import_module(module_name)
                    cls = getattr(module, type_name)
                    return cls()
                    
            # Regular dictionary
            return {k: ChompConfig._deserialize_param(v) for k, v in param.items()}
            
        # Default case
        return param
    
    @staticmethod
    def serialize(
        chefs: List[BaseChef],
        chunker: BaseChunker,
        refineries: Optional[List[BaseRefinery]] = None,
        porter: Optional[BasePorter] = None,
        handshake: Optional[BaseHandshake] = None
    ) -> Dict[str, Any]:
        """Serialize a CHOMP pipeline configuration to a dictionary.
        
        Args:
            chefs: The list of chefs in the pipeline
            chunker: The chunker used in the pipeline
            refineries: Optional refineries used in the pipeline
            porter: Optional porter used in the pipeline
            handshake: Optional handshake used in the pipeline
            
        Returns:
            Dictionary containing the serialized configuration
        """
        config = {
            "chefs": [
                {
                    "type": type(chef).__name__,
                    "module": type(chef).__module__,
                    "params": {k: ChompConfig._serialize_param(v) for k, v in chef.__dict__.items()}
                }
                for chef in chefs
            ],
            "chunker": {
                "type": type(chunker).__name__,
                "module": type(chunker).__module__,
                "params": {k: ChompConfig._serialize_param(v) for k, v in chunker.__dict__.items()}
            }
        }
        
        if refineries:
            config["refineries"] = [
                {
                    "type": type(refinery).__name__,
                    "module": type(refinery).__module__,
                    "params": {k: ChompConfig._serialize_param(v) for k, v in refinery.__dict__.items()}
                }
                for refinery in refineries
            ]
            
        if porter:
            config["porter"] = {
                "type": type(porter).__name__,
                "module": type(porter).__module__,
                "params": {k: ChompConfig._serialize_param(v) for k, v in porter.__dict__.items()}
            }
            
        if handshake:
            config["handshake"] = {
                "type": type(handshake).__name__,
                "module": type(handshake).__module__,
                "params": {k: ChompConfig._serialize_param(v) for k, v in handshake.__dict__.items()}
            }
            
        return config
    
    @staticmethod
    def _import_class(module_name: str, class_name: str) -> Type:
        """Import a class from a module.
        
        Args:
            module_name: Name of the module containing the class
            class_name: Name of the class to import
            
        Returns:
            The imported class
            
        Raises:
            ImportError: If the module or class cannot be imported
        """
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import {class_name} from {module_name}: {e}")
    
    @staticmethod
    def deserialize(config: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize a CHOMP pipeline configuration from a dictionary.
        
        Args:
            config: Dictionary containing the serialized configuration
            
        Returns:
            Dictionary containing instantiated pipeline components
            
        Raises:
            ValueError: If the config is invalid or missing required fields
            ImportError: If a component class cannot be imported
        """
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
            
        if "chunker" not in config:
            raise ValueError("Config must contain a chunker")
            
        result = {}
        
        # Deserialize chefs
        if "chefs" in config:
            chefs = []
            for chef_config in config["chefs"]:
                # Import the chef class
                chef_class = ChompConfig._import_class(
                    chef_config["module"], chef_config["type"]
                )
                
                # Create a new instance
                try:
                    # Properly deserialize parameters
                    params = {
                        k: ChompConfig._deserialize_param(v) 
                        for k, v in chef_config["params"].items()
                    }
                    
                    # For safety, filter out any params that might not be accepted
                    if hasattr(chef_class, "__init__"):
                        sig = inspect.signature(chef_class.__init__)
                        valid_params = {
                            k: v for k, v in params.items() 
                            if k in sig.parameters or k in ["_is_built"]
                        }
                    else:
                        valid_params = params
                        
                    # Create instance and set attributes
                    chef_instance = chef_class()
                    
                    # Set only the parameters that would be used by __init__
                    for k, v in valid_params.items():
                        if hasattr(chef_instance, k):
                            setattr(chef_instance, k, v)
                        
                    # Handle special cases for certain chef classes
                    if chef_class.__name__ == "MarkdownCleanerChef":
                        # Force check dependencies again
                        chef_instance._markdown_available = chef_instance._check_dependencies()
                    elif chef_class.__name__ == "HTMLCleanerChef":
                        chef_instance._bs4_available = chef_instance._check_dependencies()
                    elif chef_class.__name__ == "CSVCleanerChef":
                        chef_instance._csv_available = chef_instance._check_dependencies()
                        
                    chefs.append(chef_instance)
                except Exception as e:
                    raise ValueError(f"Failed to create chef {chef_config['type']}: {e}")
                    
            result["chefs"] = chefs
                
        # Deserialize chunker
        chunker_config = config["chunker"]
        try:
            # Import the chunker class
            chunker_class = ChompConfig._import_class(
                chunker_config["module"], chunker_config["type"]
            )
            
            # Create a new instance with proper parameter deserialization
            params = {
                k: ChompConfig._deserialize_param(v) 
                for k, v in chunker_config["params"].items()
            }
            
            # Create instance
            if chunker_class.__name__ == "RecursiveChunker":
                # For RecursiveChunker, we need special handling
                chunk_size = params.get("chunk_size", 512)
                min_characters_per_chunk = params.get("min_characters_per_chunk", 24)
                chunker_instance = chunker_class(
                    chunk_size=chunk_size,
                    min_characters_per_chunk=min_characters_per_chunk
                )
                
                # Set the rules if they were serialized
                if "rules" in params:
                    chunker_instance.rules = params["rules"]
            else:
                # For other chunker types
                chunker_instance = chunker_class()
                # Set parameter values
                for k, v in params.items():
                    if hasattr(chunker_instance, k):
                        setattr(chunker_instance, k, v)
                
            result["chunker"] = chunker_instance
        except Exception as e:
            raise ValueError(f"Failed to create chunker {chunker_config['type']}: {e}")
        
        # Deserialize refineries with similar approach
        if "refineries" in config:
            refineries = []
            for refinery_config in config["refineries"]:
                try:
                    refinery_class = ChompConfig._import_class(
                        refinery_config["module"], refinery_config["type"]
                    )
                    params = {
                        k: ChompConfig._deserialize_param(v) 
                        for k, v in refinery_config["params"].items()
                    }
                    
                    # Create a new instance
                    refinery_instance = refinery_class()
                    
                    # Set parameters
                    for k, v in params.items():
                        if hasattr(refinery_instance, k):
                            setattr(refinery_instance, k, v)
                            
                    refineries.append(refinery_instance)
                except Exception as e:
                    raise ValueError(f"Failed to create refinery {refinery_config['type']}: {e}")
                    
            result["refineries"] = refineries
                
        # Deserialize porter
        if "porter" in config:
            porter_config = config["porter"]
            try:
                porter_class = ChompConfig._import_class(
                    porter_config["module"], porter_config["type"]
                )
                params = {
                    k: ChompConfig._deserialize_param(v) 
                    for k, v in porter_config["params"].items()
                }
                
                # Create a new instance
                if porter_class.__name__ == "JSONPorter":
                    # Handle JSONPorter specially
                    lines = params.get("lines", True)
                    porter_instance = porter_class(lines=lines)
                else:
                    porter_instance = porter_class()
                
                # Set parameters
                for k, v in params.items():
                    if hasattr(porter_instance, k):
                        setattr(porter_instance, k, v)
                        
                result["porter"] = porter_instance
            except Exception as e:
                raise ValueError(f"Failed to create porter {porter_config['type']}: {e}")
            
        # Deserialize handshake
        if "handshake" in config:
            handshake_config = config["handshake"]
            try:
                handshake_class = ChompConfig._import_class(
                    handshake_config["module"], handshake_config["type"]
                )
                params = {
                    k: ChompConfig._deserialize_param(v) 
                    for k, v in handshake_config["params"].items()
                }
                
                # Create a new instance
                handshake_instance = handshake_class()
                
                # Set parameters
                for k, v in params.items():
                    if hasattr(handshake_instance, k):
                        setattr(handshake_instance, k, v)
                        
                result["handshake"] = handshake_instance
            except Exception as e:
                raise ValueError(f"Failed to create handshake {handshake_config['type']}: {e}")
            
        return result
    
    @staticmethod
    def save_json(
        config: Dict[str, Any],
        filepath: str,
        indent: int = 2
    ) -> None:
        """Save a serialized CHOMP configuration to a JSON file.
        
        Args:
            config: Serialized configuration dictionary
            filepath: Path to save the configuration to
            indent: JSON indentation level
            
        Raises:
            IOError: If the file cannot be written
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=indent, default=lambda o: str(o))
            
    @staticmethod
    def load_json(filepath: str) -> Dict[str, Any]:
        """Load a serialized CHOMP configuration from a JSON file.
        
        Args:
            filepath: Path to load the configuration from
            
        Returns:
            Serialized configuration dictionary
            
        Raises:
            IOError: If the file cannot be read
            json.JSONDecodeError: If the file contains invalid JSON
        """
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
            
    @staticmethod
    def save_pickle(config: Dict[str, Any], filepath: str) -> None:
        """Save a serialized CHOMP configuration to a pickle file.
        
        Args:
            config: Serialized configuration dictionary
            filepath: Path to save the configuration to
            
        Raises:
            IOError: If the file cannot be written
            pickle.PickleError: If the objects cannot be pickled
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, "wb") as f:
            pickle.dump(config, f)
            
    @staticmethod
    def load_pickle(filepath: str) -> Dict[str, Any]:
        """Load a serialized CHOMP configuration from a pickle file.
        
        Args:
            filepath: Path to load the configuration from
            
        Returns:
            Serialized configuration dictionary
            
        Raises:
            IOError: If the file cannot be read
            pickle.UnpicklingError: If the file contains invalid pickle data
        """
        with open(filepath, "rb") as f:
            return pickle.load(f) 