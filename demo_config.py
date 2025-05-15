#!/usr/bin/env python3
"""Demo of Chonkie's configuration system."""

import sys
import os
import json

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.chonkie import (
    Chomp,
    ChompConfig,
    CSVCleanerChef,
    HTMLCleanerChef,
    JSONCleanerChef,
    MarkdownCleanerChef, 
    RecursiveChunker,
    TextCleanerChef,
    OverlapRefinery,
    JSONPorter,
)


def demo_save_load_config():
    """Demonstrate saving and loading pipeline configurations."""
    print("\n=== Configuration System Demo ===\n")
    
    # Create a complex pipeline
    print("Creating a complex pipeline...")
    
    # Components for our pipeline
    markdown_cleaner = MarkdownCleanerChef(preserve_headings=True)
    text_cleaner = TextCleanerChef(normalize_whitespace=True, remove_urls=False)
    chunker = RecursiveChunker(chunk_size=50)  # Smaller chunk size for demonstration
    refinery = OverlapRefinery(context_size=10, method="suffix")
    
    # Note: We're not using a porter here to simplify the output
    
    # Create the pipeline
    pipeline = (
        Chomp()
        .add_chef(markdown_cleaner)
        .add_chef(text_cleaner)
        .set_chunker(chunker)
        .add_refinery(refinery)
        .build()
    )
    
    # Save the configuration to a JSON file
    print("Saving pipeline configuration to JSON...")
    pipeline.save_config("pipeline_config.json")
    
    # Load the configuration and create a new pipeline
    print("Loading pipeline configuration from JSON...")
    loaded_pipeline = Chomp.load_config("pipeline_config.json")
    
    # Print information about the loaded pipeline
    print("\nOriginal Pipeline:")
    print(pipeline)
    print("\nLoaded Pipeline:")
    print(loaded_pipeline)
    
    # Test the loaded pipeline
    markdown_content = """
    # Chonkie Configuration System Demo
    
    This is a **demonstration** of the configuration system:
    
    - Save pipeline configuration to JSON or pickle
    - Load pipeline configuration from files
    - Recreate pipelines with the same components
    
    Visit [Chonkie website](https://docs.chonkie.ai) for more information!
    """
    
    print("\nProcessing text with the loaded pipeline...")
    chunks = loaded_pipeline(markdown_content)
    
    print(f"\nProcessed text into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i + 1}: {chunk.text[:50]}...")
    
    # Clean up the config file afterwards
    try:
        os.remove("pipeline_config.json")
        print("Cleaned up configuration file.")
    except:
        pass


def demo_recipe_system():
    """Demonstrate a simple recipe system using the configuration system."""
    print("\n=== Recipe System Demo ===\n")
    
    # Define some recipe functions
    def create_markdown_recipe():
        """Create a standard markdown processing pipeline."""
        pipeline = (
            Chomp()
            .add_chef(MarkdownCleanerChef(preserve_headings=True))
            .add_chef(TextCleanerChef(normalize_whitespace=True))
            .set_chunker(RecursiveChunker())
            .build()
        )
        return pipeline.to_config()
    
    def create_html_recipe():
        """Create a standard HTML processing pipeline."""
        try:
            pipeline = (
                Chomp()
                .add_chef(HTMLCleanerChef(preserve_line_breaks=True))
                .add_chef(TextCleanerChef(normalize_whitespace=True, remove_urls=True))
                .set_chunker(RecursiveChunker())
                .build()
            )
            return pipeline.to_config()
        except ValueError:
            print("Warning: HTMLCleanerChef not available, skipping HTML recipe")
            return None
    
    def create_json_recipe():
        """Create a standard JSON processing pipeline."""
        pipeline = (
            Chomp()
            .add_chef(JSONCleanerChef(flatten=True))
            .set_chunker(RecursiveChunker())
            .build()
        )
        return pipeline.to_config()
    
    # Create a recipe catalog
    recipes = {
        "markdown": create_markdown_recipe(),
        "json": create_json_recipe()
    }
    
    # Add HTML recipe if available
    html_recipe = create_html_recipe()
    if html_recipe:
        recipes["html"] = html_recipe
    
    # Save recipes to a file
    with open("recipes.json", "w") as f:
        json.dump(recipes, f, default=str, indent=2)
        
    print("Saved recipe catalog to 'recipes.json'")
    
    # Load a recipe and use it
    print("Loading the 'markdown' recipe...")
    markdown_pipeline = Chomp.from_config(recipes["markdown"])
    
    # Test the recipe
    text = "# This is a test\n\nUsing the markdown recipe from our catalog."
    chunks = markdown_pipeline(text)
    
    print(f"Processed text with the markdown recipe pipeline.")
    print(f"Resulting chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk.text[:50]}...")
        
    # Clean up the recipes file afterwards
    try:
        os.remove("recipes.json")
        print("Cleaned up recipes file.")
    except:
        pass


if __name__ == "__main__":
    print("=== CHOMP Configuration System Demo ===")
    
    try:
        demo_save_load_config()
    except Exception as e:
        print(f"Error in configuration demo: {e}")
        
    try:
        demo_recipe_system()
    except Exception as e:
        print(f"Error in recipe system demo: {e}")
    
    print("\nDemo completed!") 