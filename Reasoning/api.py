# deepseek_inference.py 调用 deepseek api 推理
import os
import json
import time
import requests
from typing import List, Optional
import logging
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI

from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("call_gpt4", "")
test_api = module_from_spec(spec)
spec.loader.exec_module(test_api)
call_api = test_api.call_gpt4

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 常量配置
MAX_RETRIES = 5
TIMEOUT_SECONDS = 300  # 3分钟超时
API_KEY = os.getenv('DEEPSEEK_API_KEY', '')


# 初始化DeepSeek客户端
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

class DeepSeekInference:
    def __init__(self):
        self.lock = Lock()  # 确保线程安全

    def deepseek_chat(self, query: str) -> Optional[str]:
        setting = [{"role": "user", "content": query}]
        retries = 0
        while retries < MAX_RETRIES:
            try:
                completion = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=setting,
                    temperature=0.5,
                    max_tokens=4096
                )
                result = completion.choices[0].message.content
                # logger.debug(f"DeepSeek inference success: {result}")
                return result
            except Exception as e:
                error_message = str(e)
                if "400" in error_message and "Content Exists Risk" in error_message:
                    logger.error(f"Bad request error: {error_message}. Skipping this query.")
                    return None
                retries += 1
                logger.warning(f"Request failed: {error_message}. Retrying in 10 seconds... ({retries}/{MAX_RETRIES})")
                time.sleep(10)
        logger.error("Max retries reached for query. Skipping.")
        return None

    def run_inference(self, prompt: str) -> Optional[str]:
        """
        单个推理请求，符合model_inference.py中的接口。
        """
        try:
            start_time = time.time()
            response = self.deepseek_chat(prompt)

            if response and (time.time() - start_time) > TIMEOUT_SECONDS:
                logger.error("Processing exceeded 3 minutes timeout.")
                return None

            return response.strip() if response else None
        except Exception as e:
            logger.error(f"Error in run_inference: {str(e)}")
            return None

    def run_batch_inference(self, prompts: List[str]) -> List[Optional[str]]:
        """批量推理请求，符合model_inference.py中的接口。

        Args:
            prompts (List[str]): 输入的提示内容列表。

        Returns:
            List[Optional[str]]: 推理结果列表，对应每个输入提示。
        """
        results = [None] * len(prompts)
        max_workers = 48  # 根据需要调整最大线程数

        def process_prompt(index_prompt):
            index, prompt = index_prompt
            result = self.run_inference(prompt)
            return index, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = executor.map(process_prompt, enumerate(prompts))
            for index, result in futures:
                results[index] = result

        # 确保所有结果都有值
        results = [res if res is not None else "Error: Inference failed" for res in results]
        return results

# 全局DeepSeekInference实例
_model_instance = None

def initialize_model(model_path: str = None, gpu_id: str = '0', batch_size: int = 32):
    """初始化全局DeepSeekInference实例
    Args:
        model_path (str, optional): 模型路径。对于DeepSeek API，此参数可以忽略或设置为None。
        gpu_id (str, optional): GPU ID。对于DeepSeek API，此参数可以忽略。
        batch_size (int, optional): 批处理大小。对于DeepSeek API，此参数可以忽略或设置为默认值。
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = DeepSeekInference()
    return _model_instance

def get_model_inference():
    """获取全局DeepSeekInference实例"""
    if _model_instance is None:
        raise RuntimeError("Model not initialized. Call initialize_model first.")
    return _model_instance

def run_gpt_4o_inference(prompt: str) -> Optional[str]:
    try:
        # 正确调用函数而不是用字典索引
        result = call_api(prompt)
        return result[0] if isinstance(result, (list, tuple)) else result
    except Exception as e:
        logging.error(f"GPT-4推理失败: {str(e)}")
        return None

def run_gpt_4o_batch_inference(prompts: List[str]) -> List[Optional[str]]:
    results = [None] * len(prompts)
    max_workers = 24
    def process_prompt(index_prompt):
        index, prompt = index_prompt
        result = run_gpt_4o_inference(prompt)
        return index, result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(process_prompt, enumerate(prompts))
        for index, result in futures:
            results[index] = result
    results = [res if res is not None else "Error: Inference failed" for res in results]
    return results

def qwq(prompt):
    # Internal configuration
    api_key = ""  # Replace with your actual API key
    max_retries = 100
    url = "https://cn2us02.opapi.win/v1/chat/completions"

    payload = json.dumps({
        "model": "ark-deepseek-r1-250120",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

    headers = {
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json',
        "Authorization": 'Bearer ' + api_key,
    }

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            res = response.json()

            if 'choices' in res and len(res['choices']) > 0:
                res_content = res['choices'][0]['message']['content']
                return res_content
            else:
                error_message = f"Unexpected response structure: {res}"
                print(error_message)
                return None

        except requests.exceptions.RequestException as e:
            error_message = f"Request failed: {str(e)}"
            print(f"{error_message}. Retrying in 10 seconds... (Attempt {retries + 1}/{max_retries})")
            time.sleep(3)
            retries += 1

        except json.JSONDecodeError as e:
            error_message = f"Failed to decode JSON response: {str(e)}"
            print(error_message)
            return None

    print(f"Maximum retries reached. Failed to process prompt: ")
    return None

def run_qwq_inference(prompt: str) -> Optional[str]:
    try:
        # 正确调用函数而不是用字典索引
        result = qwq(prompt)
        return result if isinstance(result, (list, tuple)) else result
    except Exception as e:
        logging.error(f"GPT-4推理失败: {str(e)}")
        return None

def run_qwq_batch_inference(prompts: List[str]) -> List[Optional[str]]:
    results = [None] * len(prompts)
    max_workers = 16
    def process_prompt(index_prompt):
        index, prompt = index_prompt
        result = run_qwq_inference(prompt)
        return index, result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(process_prompt, enumerate(prompts))
        for index, result in futures:
            results[index] = result
    results = [res if res is not None else "Error: Inference failed" for res in results]
    return results

def run_inference(prompt: str) -> Optional[str]:
    """运行单条推理的全局函数"""
    model = get_model_inference()
    result = model.run_inference(prompt)
    # logger.debug(f"Inference result for prompt: {prompt} -> {result}")
    return result

def run_batch_inference(prompts: List[str]) -> List[Optional[str]]:
    """运行批量推理的全局函数"""
    model = get_model_inference()
    results = model.run_batch_inference(prompts)
    for i, (prompt, result) in enumerate(zip(prompts, results)):
        logger.debug(f"Batch inference - Prompt {i}: {prompt} -> {result}")
    return results

# 测试代码
if __name__ == "__main__":
    # Initialize model
    initialize_model()
    prompts = ["你好"] * 100

    # Test 1: Individual requests with threading
    print("\nTesting 20 threaded individual requests:")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=48) as executor:
        futures = []
        for prompt in prompts:
            future = executor.submit(run_inference, prompt)
            futures.append(future)

        results_threaded = []
        for future in tqdm(futures, total=len(futures), desc="Individual requests"):
            try:
                result = future.result()
                results_threaded.append(result)
            except Exception as e:
                print(f"Error in inference: {str(e)}")
                results_threaded.append(None)

    threaded_time = time.time() - start_time
    successful_threaded = len([r for r in results_threaded if r is not None])

    # Test 2: Batch inference
    print("\nTesting batch inference:")
    start_time = time.time()
    results_batch = run_batch_inference(prompts)
    batch_time = time.time() - start_time
    successful_batch = len([r for r in results_batch if r != "Error: Inference failed"])

    # Print comparison
    print("\n=== Performance Comparison ===")
    print(f"\nThreaded Individual Requests:")
    print(f"Time taken: {threaded_time:.2f} seconds")
    print(f"Successful requests: {successful_threaded}/100")
    print(f"Average time per request: {threaded_time / 100:.2f} seconds")

    print(f"\nBatch Inference:")
    print(f"Time taken: {batch_time:.2f} seconds")
    print(f"Successful requests: {successful_batch}/100")
    print(f"Average time per request: {batch_time / 100:.2f} seconds")

    print(f"\nTime Difference:")
    print(
        f"Batch is {(threaded_time / batch_time):.2f}x {'faster' if batch_time < threaded_time else 'slower'} than individual requests")
