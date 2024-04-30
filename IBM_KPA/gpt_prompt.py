import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import time
  
def get_rationale(sentence):
    # The API endpoint and key placeholders
    load_dotenv()
    client = OpenAI(api_key=os.environ.get("API_KEY"))

    prompt = f"Provide the possible phrase from a sentence, which could decide the stance or label of the sentence. The phrase should only be original phrase of the sentence.\nSentence:{sentence}" 
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_tokens=150
    )

    r = response.choices[0].message.content
    return r.rstrip("\"")

def process_chunk(chunk):
    chunk['rationale'] = chunk['key_point'].apply(get_rationale)
    return chunk

def split_process_combine(input_filepath, output_filepath):
    # Read the CSV in chunks
    chunk_size = 5000
    chunks = pd.read_csv(input_filepath, chunksize=chunk_size)
    
    processed_chunks = []
    
    # Process each chunk and pause for 1 minute between each
    for chunk in chunks:
        processed_chunk = process_chunk(chunk)
        processed_chunks.append(processed_chunk)
        print("Processed a chunk, waiting for 1 minute before processing the next one.")
        time.sleep(60)  # Wait for 60 seconds

    # Combine all processed chunks
    final_df = pd.concat(processed_chunks)
    final_df.to_csv(output_filepath, index=False)
    print("All chunks processed and combined. Output file saved.")


input_filepath = 'data\ArgKP-2021_dataset.csv'
output_filepath = 'data\ArgKP-2021_dataset_r.csv'
split_process_combine(input_filepath, output_filepath)
