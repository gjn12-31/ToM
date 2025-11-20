import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import copy
import torch
import re
import tiktoken
import time
import numpy as np
from typing import List, Dict, Any
from FlagEmbedding import FlagModel
from sklearn.metrics.pairwise import cosine_similarity
from prompts import prefix
from util import *
from tqdm import tqdm
from openai import OpenAI


# Initialize BGE model for embeddings
bge_model = FlagModel("BAAI/bge-small-en-v1.5", use_fp16=True)


def get_bge_embedding(text: str) -> np.ndarray:
    text = text.replace("\n", " ").strip()
    embedding = bge_model.encode([text])
    return embedding[0]


def filter_blocks_by_similarity(blocks: List[str], question: str, top_k: int = 30) -> List[str]:
    """Filter text blocks by similarity to the question, with progress bar."""
    if len(blocks) <= top_k:
        print(f"Total blocks ({len(blocks)}) are fewer than top_k ({top_k}), returning all blocks")
        return blocks

    print(f"Computing embeddings for question and {len(blocks)} blocks...")
    question_embedding = get_bge_embedding(question)

    # Add progress bar for embedding computation
    block_embeddings = []
    for block in tqdm(blocks, desc="Computing block embeddings"):
        block_embeddings.append(get_bge_embedding(block))

    print("Computing similarities...")
    similarities = cosine_similarity([question_embedding], block_embeddings)[0]

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    filtered_blocks = [blocks[i] for i in top_indices]

    # Print top similarity scores for transparency
    top_scores = [similarities[i] for i in top_indices]
    print(f"Top similarity scores: {[round(score, 3) for score in top_scores[:5]]}...")
    print(f"Filtered blocks from {len(blocks)} to {len(filtered_blocks)} (keeping top {top_k} blocks)")
    return filtered_blocks


# Global constants
RANDOM_SEED = 224
API_KEY = os.getenv('DEEPSEEK_API_KEY', '')  # Get API key from environment

# Initialize DeepSeek client
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)


def chat(query: str) -> str:
    """Call LLM API to generate summary"""
    setting = [{"role": "user", "content": query}]

    # Initialize retry counter
    retry_count = 0
    max_retries = 10

    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model="deepseek-chat",
                messages=setting,
                temperature=0.0,
                max_tokens=4096
            )
            # Return just the content, not a list
            return completion.choices[0].message.content
        except Exception as e:
            retry_count += 1
            remaining = max_retries - retry_count

            if retry_count >= max_retries:
                print(f"Maximum retry count reached ({max_retries}), request failed: {str(e)}")
                return f"Request failed, maximum retry count reached: {str(e)}"

            print(f"Request failed: {str(e)}. Retrying in 10 seconds... (Remaining retries: {remaining})")
            time.sleep(10)


def run_inference(blocks):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    results = []
    results_lock = threading.Lock()

    # Add progress tracking
    total_blocks = len(blocks)
    processed_blocks = 0
    progress_lock = threading.Lock()

    def process_block(block, block_idx):
        nonlocal processed_blocks

        try:
            # Build request
            request = {
                'input_block': block,
                'block_index': block_idx
            }

            # API call - fixed to pass just the string, not wrap it in a list
            response = chat(prefix + block)
            # response, _ = call_gpt4(prefix + block)

            # Thread-safe result saving
            with results_lock:
                result_item = copy.deepcopy(request)
                result_item["generated_response"] = response
                results.append(result_item)

            # Update progress
            with progress_lock:
                processed_blocks += 1
                if processed_blocks % 5 == 0 or processed_blocks == total_blocks:
                    print(
                        f"Progress: {processed_blocks}/{total_blocks} blocks processed ({processed_blocks / total_blocks * 100:.1f}%)")

            time.sleep(0.1)  # Keep original rate limiting

        except Exception as e:
            print(f"Error processing block {block_idx}: {str(e)}")
            # Still count as processed for progress tracking
            with progress_lock:
                processed_blocks += 1
            return None

    # Use thread pool for parallel processing
    print(f"Starting inference on {total_blocks} blocks with 24 workers...")
    with ThreadPoolExecutor(max_workers=24) as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_block, block, idx)
            for idx, block in enumerate(blocks)
        ]

        # Wait for all tasks to complete
        for future in as_completed(futures):
            try:
                future.result()  # Get result, mainly to catch potential exceptions
            except Exception as e:
                print(f"Unexpected error in thread: {str(e)}")

    # Sort results by block_index
    results.sort(key=lambda x: x['block_index'])
    return results


def save_structured_blocks(inference_results, current_id, question, answer, output_dir):
    structured_data = []

    for block_id, result in enumerate(inference_results):
        block_data = {
            "id": current_id,
            "block_id": block_id,  # Add block_id
            "question": question,
            "answer": answer,
            "input_block": result.get('input_block', ''),
            "output_block": result.get('generated_response', '')
        }
        structured_data.append(block_data)

    # Append to JSON file
    output_file = os.path.join(output_dir, 'structured_blocks.json')

    # If file exists, read existing data
    existing_data = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []

    # Merge existing and new data
    existing_data.extend(structured_data)

    # Save merged data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)


# Function to save results for a single item to each top_k JSON file
def save_single_result(item_results, output_dir, current_id, top_k_values):
    """Save results for a single processed item to each top_k JSON file."""
    for top_k, result in item_results.items():
        if not result:  # Skip if result is empty
            continue

        output_file = os.path.join(output_dir, f'processed_results_topk_{top_k}.json')

        # If file exists, load existing data
        existing_data = []
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                # If file is empty or invalid JSON, start with empty list
                existing_data = []
                print(f"Warning: Could not load existing data from {output_file}, starting with empty list")

        # Append new result
        existing_data.append(result)

        # Save updated data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

        print(f"Result for item ID {current_id} saved to {output_file} (top_k={top_k})")


def process_text_file_multi_topk_optimized(input_file: str, output_dir: str, top_k_values: List[int], start_index=0,
                                           end_index=100):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output files will be saved to directory: {output_dir}")

    # Sort top_k values to process efficiently
    sorted_top_k = sorted(top_k_values)
    max_top_k = sorted_top_k[-1]
    print(f"Processing for top_k values: {sorted_top_k} (optimized)")

    # Load data first
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data_list = [json.loads(line.strip()) for line in f][0:50]

    # Slice data if needed
    if end_index > 0:
        data_list = data_list[start_index:end_index]
        print(f"Processing items from index {start_index} to {end_index - 1}")
    else:
        data_list = data_list[start_index:]
        print(f"Processing all items starting from index {start_index}")

    total_items = len(data_list)

    # Process each item with progress tracking
    for item_idx, data in enumerate(data_list):
        print(f"\n--- Processing item {item_idx + 1}/{total_items} ---")

        # Initialize results for this item across all top_k values
        item_results = {k: None for k in top_k_values}

        if isinstance(data, dict):
            context = data.get('context', '')
            current_id = data.get('_id')
            question = data.get('input', '')
            answer = data.get('answers', '')

            if not context or current_id is None:
                print(f"Warning: Row missing required fields 'context' or 'id'")
                continue

            try:
                # Get initial blocks
                print(f"Splitting text into blocks for ID {current_id}...")
                initial_blocks = split_text_into_token_blocks(context)
                print(f"Initial blocks for ID {current_id}: {len(initial_blocks)}")

                # Filter blocks by similarity
                filtered_blocks = filter_blocks_by_similarity(initial_blocks, question, max_top_k)

                # Save original blocks for debugging
                debug_blocks_file = os.path.join(output_dir, f'debug_blocks_{current_id}.json')
                with open(debug_blocks_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "id": current_id,
                        "original_blocks": initial_blocks,
                        "filtered_blocks": filtered_blocks,
                        "question": question
                    }, f, ensure_ascii=False, indent=2)
                print(f"Saved debug blocks to {debug_blocks_file}")

                # Run inference
                print(f"Running inference on {len(filtered_blocks)} blocks...")
                inference_results = run_inference(blocks=filtered_blocks)

                # Save structured blocks
                save_structured_blocks(inference_results, current_id, question, answer, output_dir)

                # Store all trees and original blocks
                all_trees = []
                original_blocks = []

                print("Parsing tree responses...")
                for response in tqdm(inference_results, desc="Parsing responses"):
                    try:
                        block_idx = response.get('block_index', -1)
                        # Debug output for parsing issues
                        generated_text = response.get('generated_response', '')
                        if generated_text.startswith("* 警告") or generated_text.startswith("错误"):
                            print(f"Warning: Potentially problematic response for block {block_idx}:")
                            print(generated_text[:200] + "..." if len(generated_text) > 200 else generated_text)

                        result = parse_tree_response(response, block_idx)
                        if result:
                            tree, _ = result  # Correctly unpack tuple
                            if tree:  # tree is a list
                                for t in tree:  # Iterate through each tree in the list
                                    if isinstance(t, dict):  # Ensure it's a dictionary
                                        t['input_block'] = response.get('input_block', '')
                                        t['block_index'] = block_idx
                                        all_trees.append(t)
                                        original_blocks.append(response.get('input_block', ''))
                    except Exception as e:
                        print(f"Error processing tree for block {block_idx}: {str(e)}")
                        print(f"Response preview: {str(response.get('generated_response', ''))[:200]}...")
                        continue

                # Process each top_k value
                print(f"Merging trees for each top_k value...")
                for top_k in sorted_top_k:
                    try:
                        relevant_trees = all_trees[:top_k]
                        if relevant_trees:
                            merged_tree = merge_trees(relevant_trees)
                            if merged_tree:
                                output_entry = {
                                    "id": current_id,
                                    "question": question,
                                    "answer": answer,
                                    "trees": merged_tree,
                                    "original_blocks": original_blocks[:top_k]  # Save original text blocks
                                }
                                # Store result for this top_k
                                item_results[top_k] = output_entry
                                print(f"Successfully processed for top_k={top_k}")
                            else:
                                print(f"Warning: Failed to merge trees for top_k={top_k}")
                        else:
                            print(f"Warning: No relevant trees found for top_k={top_k}")
                    except Exception as e:
                        print(f"Error processing top_k={top_k}: {str(e)}")
                        continue

                # Save results for this item immediately
                save_single_result(item_results, output_dir, current_id, top_k_values)
                print(f"Results for item ID {current_id} saved to disk")

            except Exception as e:
                print(f"Processing error for item {item_idx + 1}: {str(e)}")
                continue
        else:
            print(f"Warning: Row is not a valid JSON object")

    print(f"Processing complete. Results saved to {output_dir}")


def test_chat_function():
    """测试chat函数是否正常工作"""
    print("\n===== 测试 chat() 函数 =====")
    test_query = "请生成一个简单的树状结构来表示以下文本：'这是一个测试文本，用来检测API调用是否正常工作。'"
    print("发送测试查询...")
    try:
        # response, _ = call_gpt4(test_query)
        response = chat(test_query)
        print("API响应成功!")
        print("响应前100个字符:")
        print(response[:100] + "..." if len(response) > 100 else response)
        return True
    except Exception as e:
        print(f"API测试失败! 错误: {str(e)}")
        return False


if __name__ == "__main__":
    start_time = time.time()
    input_file = ''
    output_dir = ''
    top_k_values = [3]

    # 在主流程开始前测试chat()函数
    api_test_success = test_chat_function()
    if not api_test_success:
        print("警告：API测试失败，请检查API设置后重试")
        # 可选：如果API测试失败，可以选择退出程序
        # import sys
        # sys.exit(1)

    print("\n===== 开始主流程 =====")
    # Add options to process a subset of the data
    start_index = 0  # Start from the first item
    end_index = 100  # Process the first 100 items (set to 0 to process all)

    process_text_file_multi_topk_optimized(input_file, output_dir, top_k_values, start_index, end_index)

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Format output runtime, convert to hours:minutes:seconds
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
