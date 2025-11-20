import json
import asyncio
import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from api import run_inference, run_batch_inference, run_gpt_4o_inference, run_gpt_4o_batch_inference, run_qwq_inference, run_qwq_batch_inference
from prompts import format_map_prompt, format_reduce_prompt
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """文档树节点数据结构"""
    title: str
    keywords: List[str]
    summary: str
    original_text: str
    children: List['TreeNode']


class AsyncMapReducePipeline:
    def __init__(self, map_prompt: str, reduce_prompt: str, base_workers: int = 4):
        self.map_prompt = map_prompt
        self.reduce_prompt = reduce_prompt
        self.base_workers = base_workers
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)
        self.cpu_count = os.cpu_count() or 4
        self.logger = logger
        self.total_nodes = 0
        self.processed_nodes = 0
        self.current_batch = 0
        self.total_batches = 0

    def format_node_content(self, node: TreeNode) -> str:
        """格式化节点信息"""
        if len(node.keywords) == 1 and ';' in node.keywords[0]:
            keywords = [k.strip() for k in node.keywords[0].split(';')]
        else:
            keywords = node.keywords

        return f"title：{node.title}\nkeywords：{', '.join(keywords)}\nsummary：{node.summary}\noriginal_text：{node.original_text}"

    async def process_nodes_parallel(self, nodes: List[TreeNode], question: str) -> List[Optional[str]]:
        """并行处理同层节点"""
        if not nodes:
            return []

        current_layer_size = len(nodes)
        concurrency = min(current_layer_size, self.cpu_count * 2)
        semaphore = asyncio.Semaphore(concurrency)

        # 处理子节点
        child_results_tasks = []
        for node in nodes:
            if node.children:
                task = asyncio.create_task(self.process_nodes_parallel(node.children, question))
                child_results_tasks.append(task)
            else:
                child_results_tasks.append(asyncio.create_task(asyncio.sleep(0)))

        child_results_list = await asyncio.gather(*child_results_tasks)

        # Reduce阶段
        reduced_results = []
        for child_results in child_results_list:
            if child_results and any(r is not None for r in child_results):
                reduced_result = await self.reduce_parallel(
                    [r for r in child_results if r is not None],
                    question
                )
                reduced_results.append(reduced_result)
            else:
                reduced_results.append(None)

        # Map阶段
        map_results = await self.map_batch_parallel(nodes, question, reduced_results, semaphore)
        return map_results

    async def map_batch_parallel(self,
                               nodes: List[TreeNode],
                               question: str,
                               child_results: List[Optional[str]],
                               semaphore: asyncio.Semaphore) -> List[Optional[str]]:
        """并行执行map操作"""
        try:
            # 准备prompts
            prompts = []
            prompt_indices = []
            for i, node in enumerate(nodes):
                node_content = self.format_node_content(node)
                if len(node.children) == 0:
                    prompt = format_map_prompt(question, node_content, "")
                else:
                    child_result = child_results[i] if child_results else None
                    if child_result is None:
                        prompts.append(None)
                        continue
                    prompt = format_map_prompt(question, node_content, child_result)

                if prompt is not None:
                    prompts.append(prompt)
                    prompt_indices.append(i)

            valid_prompts = [p for p in prompts if p is not None]
            if not valid_prompts:
                return [None] * len(nodes)

            # 批次处理
            batch_size = 32
            self.total_batches = (len(valid_prompts) + batch_size - 1) // batch_size
            self.current_batch = 0

            async def process_batch(batch_prompts: List[str]) -> List[str]:
                async with semaphore:
                    try:
                        self.current_batch += 1
                        loop = asyncio.get_event_loop()
                        results = await loop.run_in_executor(
                            self.executor,
                            run_batch_inference,
                            batch_prompts
                        )
                        self.processed_nodes += len(batch_prompts)
                        self.logger.info(
                            f"进度: {self.processed_nodes}/{self.total_nodes} 节点 | "
                            f"当前批次: {self.current_batch}/{self.total_batches}")
                        return results
                    except Exception as e:
                        self.logger.error(f"批次 {self.current_batch} 处理失败: {str(e)}")
                        return [None] * len(batch_prompts)

            # 创建批次任务
            batch_tasks = []
            for i in range(self.total_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(valid_prompts))
                batch_prompts = valid_prompts[start_idx:end_idx]
                task = asyncio.create_task(process_batch(batch_prompts))
                batch_tasks.append(task)

            batch_results = await asyncio.gather(*batch_tasks)

            # 展平并还原结果顺序
            flat_results = []
            for batch in batch_results:
                if batch:
                    flat_results.extend(batch)

            final_results = [None] * len(nodes)
            for idx, result in zip(prompt_indices, flat_results):
                final_results[idx] = result

            return final_results

        except Exception as e:
            self.logger.error(f"Map批处理失败: {str(e)}")
            return [None] * len(nodes)

    async def reduce_parallel(self, results: List[str], question: str) -> Optional[str]:
        """异步执行reduce操作"""
        if not results:
            return None
        if len(results) == 1:
            return results[0]

        # 修改结果组织方式，使用 chunk[index]:content 的格式
        results_str = ';'.join([f"chunk[{i}]:{result}" for i, result in enumerate(results)])

        prompt = format_reduce_prompt(question, results_str)
        loop = asyncio.get_event_loop()

        try:
            result = await loop.run_in_executor(self.executor, run_inference, prompt)
            return result
        except Exception as e:
            self.logger.error(f"Reduce处理失败: {str(e)}")
            return None

    async def process_tree(self, root: TreeNode, question: str) -> Optional[str]:
        """处理整个文档树"""
        def count_nodes(node: TreeNode) -> int:
            return 1 + sum(count_nodes(child) for child in node.children)

        self.total_nodes = count_nodes(root)
        self.processed_nodes = 0
        self.logger.info(f"开始处理 | 总节点数: {self.total_nodes}")

        results = await self.process_nodes_parallel([root], question)
        self.logger.info("处理完成")
        return results[0] if results else None


def create_tree_node(data: Dict[str, Any]) -> TreeNode:
    """从字典创建TreeNode对象"""
    return TreeNode(
        title=data['title'],
        keywords=data['keywords'],
        summary=data['summary'],
        original_text=data['original_text'],
        children=[create_tree_node(child) for child in data.get('children', [])]
    )