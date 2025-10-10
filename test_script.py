# test_script.py

import time
from chonkie.types import Chunk
from chonkie.handshakes import ChromaHandshake

def run_test():
    print("--- Starting ChromaHandshake Test ---")

    # 1. Create some sample data as Chunk objects
    print("1. Creating sample chunks...")
    chunks_to_add = [
        Chunk(text="The sky is blue during a clear day."),
        Chunk(text="Apples are a type of fruit that grow on trees."),
        Chunk(text="Python is a popular programming language."),
        Chunk(text="The ocean is a vast body of salt water."),
    ]
    
    # 2. Initialize the ChromaHandshake
    #    This will create a temporary, in-memory ChromaDB collection.
    print("2. Initializing ChromaHandshake...")
    # We give it a specific name to avoid randomness for the test
    handshake = ChromaHandshake(collection_name="my-test-collection")
    print(f"   Handshake created for collection: '{handshake.collection_name}'")

    # 3. Write the chunks to the database
    print("\n3. Writing chunks to the collection...")
    handshake.write(chunks_to_add)
    
    # Give Chroma a moment to index the data
    time.sleep(1) 

    # 4. Use the search method you fixed!
    print("\n4. Performing search...")
    search_query = "What language should I learn to code?"
    print(f"   Searching for: '{search_query}'")
    
    search_results = handshake.search(query=search_query, top_k=2)

    # 5. Print the results and verify
    print("\n5. Verifying results...")
    if not search_results:
        print("   ERROR: No results returned!")
        return

    print(f"   Found {len(search_results)} result(s):")
    for i, chunk in enumerate(search_results):
        print(f"   Result {i+1}: '{chunk.text}'")

    # Check if the top result is the one we expect
    expected_result = "Python is a popular programming language."
    if search_results[0].text == expected_result:
        print("\n✅ SUCCESS: The top search result is correct!")
    else:
        print(f"\n❌ FAILURE: Expected '{expected_result}' but got '{search_results[0].text}'")

    print("\n--- Test Finished ---")

if __name__ == "__main__":
    run_test()
    
#Fouzhan: Below is the content of src/chonkie/handshakes/base.py for reference.