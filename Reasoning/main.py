# 主函数
import time
import json
import asyncio
import logging
from typing import Dict, Any, List
from pipeline import AsyncMapReducePipeline, create_tree_node
from api import initialize_model
from prompts import get_prompts

# 配置日志
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)


async def process_single_record(
        record: Dict[str, Any],
        model_path: str = None,
        max_workers: int = 4
) -> Dict[str, Any]:

    initialize_model(model_path=model_path)

    prompts = get_prompts()

    pipeline = AsyncMapReducePipeline(
        map_prompt=prompts['map_prompt'],
        reduce_prompt=prompts['reduce_prompt'],
        base_workers=max_workers
    )

    tree_data = record['trees']
    root = create_tree_node(tree_data)
    question = record['question']

    try:
        raw_prediction = await pipeline.process_tree(root, question)

        key_info = extract_between(raw_prediction,
                                   "**key_info**",
                                   "**reasoning_process**")

        reasoning_process = extract_between(raw_prediction,
                                            "**reasoning_process**",
                                            "**answer**")

        answer = extract_between(raw_prediction,
                                 "**answer**",
                                 "**confidence_score**")

        confidence_score = extract_between(raw_prediction,
                                           "**confidence_score**",
                                           "***end output***")

        result = {
            'id': record['id'],
            'question': question,
            'answer': record['answer'],
            'prediction': answer,
            'structured_info': {
                'key_info': key_info,
                'reasoning_process': reasoning_process,
                'answer': answer,
                'confidence_score': confidence_score
            }
        }
        return result
    except Exception as e:
        print(f"Error processing record {record['id']}: {str(e)}")
        raise


def extract_between(text: str, start_marker: str, end_marker: str) -> str:
    if start_marker not in text:
        return ""

    start_idx = text.index(start_marker) + len(start_marker)

    if end_marker is None:
        return text[start_idx:].strip()

    if end_marker not in text:
        return ""

    end_idx = text.index(end_marker)
    return text[start_idx:end_idx].strip()


async def process_all_records(
        input_path: str,
        output_path: str,
        model_path: str = None,
        max_workers: int = 4,
        max_concurrent: int = 10
) -> None:
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            records = json.load(f)[0:50]
        logger.info(f"Found {len(records)} records to process")

        semaphore = asyncio.Semaphore(max_concurrent)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('[\n')

        processed_count = 0

        async def process_with_semaphore(record):
            nonlocal processed_count
            async with semaphore:
                try:
                    result = await process_single_record(
                        record=record,
                        model_path=model_path,
                        max_workers=max_workers
                    )

                    if result is not None:
                        with open(output_path, 'a', encoding='utf-8') as f:
                            if processed_count > 0:
                                f.write(',\n')
                            json_str = json.dumps(result, ensure_ascii=False, indent=2)
                            f.write(json_str)

                        processed_count += 1
                        logger.info(f"Processed and saved record {processed_count}/{len(records)}")

                    return result
                except Exception as e:
                    logger.error(f"Failed to process record {record['id']}: {str(e)}")
                    return None

        tasks = [process_with_semaphore(record) for record in records]

        results = await asyncio.gather(*tasks)

        with open(output_path, 'a', encoding='utf-8') as f:
            f.write('\n]')

        valid_results = [result for result in results if result is not None]
        logger.info(f"Processing complete. Successfully processed {len(valid_results)}/{len(records)} records.")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        try:
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write('\n]')
        except:
            pass
        raise


async def main():
    config = {
        'input_path': '',
        'output_path': '',
        'model_path': None,
        'max_workers': 4,
        'max_concurrent': 10
    }

    start_time = time.time()
    try:
        await process_all_records(**config)
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")

    end_time = time.time()
    print(f"total time is {end_time - start_time}")


if __name__ == "__main__":
    asyncio.run(main())