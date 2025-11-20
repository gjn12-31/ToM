import os
import numpy as np
import pandas as pd
import json
import time
import warnings
import multiprocessing
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

import umap.umap_ as umap
# from scipy import spatial
from sklearn.mixture import GaussianMixture
from FlagEmbedding import FlagModel
from openai import OpenAI
from transformers import logging

# 设置警告和日志过滤
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="You're using a BertTokenizerFast tokenizer")
logging.set_verbosity_error()

# 导入外部API调用模块
# from importlib.util import spec_from_file_location, module_from_spec


# 全局常量
RANDOM_SEED = 224
API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
# 填写 deepseek api key

# 初始化DeepSeek客户端
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# 摘要模板
SUMMARY_TEMPLATE = """
You are a senior text analysis and summary expert. Please conduct an in-depth analysis of the input text passages and generate a structured comprehensive summary report.

Input:
{context} // The text passages to be analyzed will be inserted here

Please output strictly according to the following format:

**Title**: [Concisely and accurately summarize the core theme of the text, maximum 15 words]

**Keywords**: [Extract 3-5 most representative keywords, separated by semicolons (;), each keyword 2-4 words]

**Summary**: [Use objective and accurate language to summarize the core content and main points in several complete sentences, total word count within 200 words]

Notes:
1. Strictly follow the output format above
2. Ensure all three sections (Title/Keywords/Summary) are complete
3. Maintain professional and objective expression
4. Do not add any additional explanations or commentary
"""


# 单例模式获取嵌入模型
def get_embedder():
    """获取嵌入模型的单例"""
    if not hasattr(get_embedder, '_embedder'):
        # 添加线程锁防止并发初始化
        with multiprocessing.Lock():
            if not hasattr(get_embedder, '_embedder'):
                print("初始化嵌入模型...")
                get_embedder._embedder = FlagModel("BAAI/bge-small-en-v1.5", use_fp16=True)
                print("嵌入模型初始化完成")
    return get_embedder._embedder


def chat(query: str, max_retries=5, initial_wait=10) -> str:
    """调用LLM API生成摘要，带指数退避重试"""
    setting = [{"role": "user", "content": query}]

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="deepseek-chat",
                messages=setting,
                temperature=0.0,
                max_tokens=200
            )
            return completion.choices[0].message.content
        except Exception as e:
            wait_time = initial_wait * (2 ** attempt)  # 指数退避
            print(f"API请求失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}. {wait_time}秒后重试...")
            time.sleep(wait_time)

    # 所有重试都失败后返回空结果
    print("所有API请求尝试都失败，返回空结果")
    return ""


def global_cluster_embeddings(
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
        min_dist: float = 0.1,
) -> np.ndarray:
    """使用UMAP进行全局维度降低"""
    if n_neighbors is None:
        n_neighbors = max(3, min(int(len(embeddings) * 0.1), 15))

    return umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=dim,
        min_dist=min_dist,
        metric=metric,
        random_state=RANDOM_SEED
    ).fit_transform(embeddings)


def local_cluster_embeddings(
        embeddings: np.ndarray,
        dim: int,
        num_neighbors: int = 10,
        metric: str = "cosine"
) -> np.ndarray:
    """使用UMAP进行局部维度降低"""
    return umap.UMAP(
        n_neighbors=num_neighbors,
        n_components=dim,
        metric=metric,
        random_state=RANDOM_SEED
    ).fit_transform(embeddings)


def get_optimal_clusters(
        embeddings: np.ndarray,
        max_clusters: int = 50,
        min_clusters: int = 2,
        random_state: int = RANDOM_SEED
) -> int:
    """确定最佳聚类数量"""
    # 确保聚类数量合理
    sample_count = len(embeddings)
    if sample_count < 4:
        return 2

    max_clusters = min(max_clusters, sample_count // 2)

    n_clusters = np.arange(min_clusters, max_clusters + 1)
    bics = []
    aics = []

    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
        aics.append(gm.aic(embeddings))

    # 归一化并综合考虑BIC和AIC
    if np.max(bics) > np.min(bics) and np.max(aics) > np.min(aics):
        normalized_bics = (bics - np.min(bics)) / (np.max(bics) - np.min(bics))
        normalized_aics = (aics - np.min(aics)) / (np.max(aics) - np.min(aics))
        combined_scores = normalized_bics + normalized_aics
        return n_clusters[np.argmin(combined_scores)]
    else:
        # 如果BIC或AIC不变，返回最小聚类数
        return min_clusters


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = RANDOM_SEED):
    """使用高斯混合模型进行聚类"""
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)

    # 根据概率阈值分配聚类标签
    labels = []
    for prob in probs:
        max_prob_idx = np.argmax(prob)
        if prob[max_prob_idx] > threshold:
            labels.append([max_prob_idx])
        else:
            labels.append([])  # 未超过阈值，不分配

    return labels, n_clusters


def perform_clustering(
        embeddings: np.ndarray,
        dim: int,
        threshold: float,
) -> List[np.ndarray]:
    """执行两层聚类过程"""
    # 特殊情况处理
    if len(embeddings) <= dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))]

    # 全局维度降低
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)

    # 全局聚类
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # 对每个全局聚类进行局部聚类
    for i in range(n_global_clusters):
        # 获取属于当前全局聚类的样本
        global_cluster_mask = np.array([i in gc for gc in global_clusters])
        global_cluster_embeddings_ = embeddings[global_cluster_mask]

        if len(global_cluster_embeddings_) == 0:
            continue

        if len(global_cluster_embeddings_) <= dim + 1:
            # 样本太少，简单分配
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # 局部降维与聚类
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        # 将局部聚类结果映射回原始索引
        for j in range(n_local_clusters):
            local_cluster_mask = np.array([j in lc for lc in local_clusters])
            if not any(local_cluster_mask):
                continue

            local_cluster_embeddings_ = global_cluster_embeddings_[local_cluster_mask]

            # 找到原始索引
            indices = []
            for embed in local_cluster_embeddings_:
                # 使用更高效的方式找到匹配的嵌入
                matches = np.where(np.all(embeddings == embed, axis=1))[0]
                indices.extend(matches)

            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters


def embed(texts: List[str]) -> np.ndarray:
    """生成文本嵌入"""
    embd = get_embedder()
    cleaned_texts = [text.replace("\n", " ").strip() for text in texts]
    text_embeddings = embd.encode(cleaned_texts)
    return np.array(text_embeddings)


def embed_cluster_texts(texts: List[str], threshold: float = 0.8) -> pd.DataFrame:
    """嵌入并聚类文本"""
    print(f"嵌入并聚类 {len(texts)} 个文本...")
    text_embeddings_np = embed(texts)
    cluster_labels = perform_clustering(text_embeddings_np, 10, threshold)

    df = pd.DataFrame({
        "text": texts,
        "embd": list(text_embeddings_np),
        "cluster": cluster_labels
    })

    # 打印聚类统计信息
    cluster_counts = {}
    for row in cluster_labels:
        if len(row) > 0:
            cluster_id = row[0]
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

    print(
        f"聚类结果: {len(cluster_counts)} 个聚类, {len(texts) - sum(len(c) == 0 for c in cluster_labels)}/{len(texts)} 个文本被聚类")
    return df


def extract_second_level_nodes(tree: Dict[str, Any]) -> Tuple[Dict[int, List[Dict[str, Any]]], List[str]]:
    """提取二层节点信息，按chunk_id组织，同时返回blocks列表"""
    nodes_by_chunk = {}
    blocks = []  # 存储文本块

    if not tree or "children" not in tree:
        print("警告: 输入树为空或没有子节点")
        return {}, []

    for node in tree.get("children", []):
        chunk_id = node.get("chunk_id")
        if chunk_id is None:
            print(f"警告: 发现没有chunk_id的节点: {node.get('title', '')}")
            continue

        node_info = {
            "title": node.get("title", ""),
            "keywords": node.get("keywords", []),
            "summary": node.get("summary", ""),
            "original_text": node.get("original_text", ""),
            "original_children": node.get("children", []),
            "chunk_id": chunk_id,
        }

        # 构建完整文本
        full_text = "\n".join([
            f"Title: {node_info['title']}",
            f"Keywords: {'; '.join(node_info['keywords'])}" if node_info['keywords'] else "",
            f"Summary: {node_info['summary']}",
            f"Original Text: {node_info['original_text']}" if node_info['original_text'] else ""
        ])

        node_info['full_text'] = full_text.strip()
        blocks.append(full_text.strip())

        # 按chunk_id组织节点
        if chunk_id not in nodes_by_chunk:
            nodes_by_chunk[chunk_id] = []
        nodes_by_chunk[chunk_id].append(node_info)

    print(f"提取了 {len(blocks)} 个文本块, {len(nodes_by_chunk)} 个chunk_id")
    return nodes_by_chunk, blocks


def parse_deepseek_response(response: str) -> Dict[str, Any]:
    """解析deepseek的回复为标准格式"""
    if not response or response.strip() == "":
        print("警告: 收到空响应")
        return {
            "title": "未知主题",
            "keywords": ["未分类", "未知", "自动生成"],
            "summary": "无法生成摘要，因为原始响应为空",
            "original_text": None
        }

    try:
        result = {
            "title": "",
            "keywords": [],
            "summary": "",
            "original_text": None
        }

        # 提取标题
        title_match = response.find("**Title**:")
        if title_match != -1:
            title_end = response.find("\n", title_match)
            if title_end == -1:
                title_end = len(response)
            title_text = response[title_match + 10:title_end].strip()
            if title_text and not title_text.startswith("[") and not title_text.endswith("]"):
                result["title"] = title_text

        # 提取关键词
        keywords_match = response.find("**Keywords**:")
        if keywords_match != -1:
            keywords_end = response.find("\n", keywords_match)
            if keywords_end == -1:
                keywords_end = len(response)
            keywords_text = response[keywords_match + 13:keywords_end].strip()
            if keywords_text and not keywords_text.startswith("[") and not keywords_text.endswith("]"):
                result["keywords"] = [k.strip() for k in keywords_text.split(';') if k.strip()]

        # 提取摘要
        summary_match = response.find("**Summary**:")
        if summary_match != -1:
            summary_text = response[summary_match + 12:].strip()
            if summary_text and not summary_text.startswith("[") and not summary_text.endswith("]"):
                result["summary"] = summary_text

        # 验证必要字段
        if not result["title"]:
            result["title"] = "未命名主题"
        if not result["keywords"]:
            result["keywords"] = ["未分类", "自动生成"]
        if not result["summary"]:
            result["summary"] = "未能生成有效摘要"

        return result

    except Exception as e:
        print(f"解析响应时出错: {str(e)}")
        return {
            "title": "解析错误",
            "keywords": ["错误", "解析失败", "自动生成"],
            "summary": f"解析响应时出现错误: {str(e)}",
            "original_text": None
        }


def bottom_up_clustering(nodes_by_chunk: Dict[int, List[Dict[str, Any]]],
                         blocks: List[str],
                         n_levels: int = 2,
                         threshold: float = 0.8) -> Dict[str, Any]:
    """基于原始文本块的多层自底向上聚类"""
    if not blocks:
        print("警告: 没有文本块可聚类")
        return {}

    print(f"开始自底向上聚类, 初始块数: {len(blocks)}")
    current_blocks = blocks
    current_nodes_by_chunk = nodes_by_chunk
    results = {}

    for level in range(n_levels):
        print(f"处理层级 {level}, 当前块数: {len(current_blocks)}")
        if len(current_blocks) <= 1:
            print(f"层级 {level}: 只剩1个块，停止聚类")
            break

        # 对当前层级的文本块进行聚类
        df_clusters = embed_cluster_texts(current_blocks, threshold=threshold)

        # 处理聚类结果
        clusters_dict = {}
        unclustered_chunks = []

        for idx, row in df_clusters.iterrows():
            if len(row["cluster"]) == 0:
                unclustered_chunks.append(idx)
            else:
                cluster_id = row["cluster"][0]
                if cluster_id not in clusters_dict:
                    clusters_dict[cluster_id] = []
                clusters_dict[cluster_id].append(idx)

        # 创建新的层级节点
        new_level_nodes = []
        new_blocks = []
        new_nodes_by_chunk = {}
        current_chunk_id = 0

        # 处理已聚类的块
        for cluster_id, cluster_chunks in clusters_dict.items():
            if len(cluster_chunks) > 1:
                print(f"处理聚类 {cluster_id}: 包含 {len(cluster_chunks)} 个块")
                # 收集该聚类中所有块的节点和文本
                cluster_nodes = []
                cluster_texts = []
                for chunk_id in cluster_chunks:
                    if chunk_id in current_nodes_by_chunk:
                        cluster_nodes.extend(current_nodes_by_chunk[chunk_id])
                    cluster_texts.append(current_blocks[chunk_id])

                # 合并文本并生成新节点
                formatted_txt = "\n---\n".join(cluster_texts)
                response= chat(SUMMARY_TEMPLATE.format(context=formatted_txt))
                node_info = parse_deepseek_response(response)

                new_node = {
                    "title": node_info["title"],
                    "keywords": node_info["keywords"],
                    "summary": node_info["summary"],
                    "original_text": None,
                    "children": cluster_nodes,
                    "chunk_id": current_chunk_id  # 添加chunk_id
                }
                new_level_nodes.append(new_node)

                # 保存新的文本块和节点映射
                new_blocks.append(formatted_txt)
                new_nodes_by_chunk[current_chunk_id] = [new_node]
                current_chunk_id += 1
            else:
                # 单块的簇，保留原有节点和文本
                chunk_id = cluster_chunks[0]
                if chunk_id in current_nodes_by_chunk:
                    nodes = current_nodes_by_chunk[chunk_id]
                    for node in nodes:
                        if "chunk_id" not in node:
                            node["chunk_id"] = current_chunk_id
                    new_level_nodes.extend(nodes)
                    new_blocks.append(current_blocks[chunk_id])
                    new_nodes_by_chunk[current_chunk_id] = nodes
                    current_chunk_id += 1

        # 处理未聚类的块
        if unclustered_chunks:
            print(f"层级 {level}: {len(unclustered_chunks)} 个块未被聚类")
            for chunk_id in unclustered_chunks:
                if chunk_id in current_nodes_by_chunk:
                    nodes = current_nodes_by_chunk[chunk_id]
                    for node in nodes:
                        if "chunk_id" not in node:
                            node["chunk_id"] = current_chunk_id
                    new_level_nodes.extend(nodes)
                    new_blocks.append(current_blocks[chunk_id])
                    new_nodes_by_chunk[current_chunk_id] = nodes
                    current_chunk_id += 1

        # 保存当前层级结果
        results[level] = new_level_nodes

        # 更新当前状态为新的层级
        current_blocks = new_blocks
        current_nodes_by_chunk = new_nodes_by_chunk

        # 如果没有发生任何聚类（块的数量没有减少），则提前结束
        if len(new_blocks) >= len(blocks):
            print(f"层级 {level}: 未发生聚类，块数未减少，提前结束")
            break

    print(f"聚类完成，共生成 {len(results)} 个层级")
    return results


def merge_tree_structures(original_tree: Dict[str, Any], upper_structure: Dict[str, Any]) -> Dict[str, Any]:
    """合并树结构"""
    if not upper_structure:
        print("警告: 上层结构为空，返回原始树")
        return original_tree

    # 找到最高层级
    top_level = max(upper_structure.keys())
    root_nodes = upper_structure[top_level]

    if not root_nodes:
        print("警告: 最高层级没有节点，返回原始树")
        return original_tree

    # 创建根节点
    new_root = root_nodes[0] if len(root_nodes) == 1 else {
        "title": "文档根节点",
        "keywords": ["root", "document", "collection"],
        "summary": "包含多个子主题的完整文档结构",
        "original_text": None,
        "children": root_nodes
    }

    # 递归替换临时字段
    def replace_children(node):
        if 'children' in node:
            for child in node['children']:
                if 'original_children' in child:
                    child['children'] = child['original_children']
                    # 删除临时字段
                    for field in ['full_text', 'original_children', 'chunk_id']:
                        if field in child:
                            del child[field]
                else:
                    replace_children(child)

    replace_children(new_root)
    return new_root


def process_single_item(item):
    """处理单个树项目"""
    try:
        item_id = item.get('id', '未知ID')
        print(f"开始处理项目 {item_id}")

        new_item = {key: value for key, value in item.items() if key != 'trees'}
        original_tree = item.get('trees', {})

        if not original_tree:
            print(f"项目 {item_id}: 树为空")
            new_item['trees'] = {}
            return new_item

        # 提取二级节点和文本块
        second_level_nodes, blocks = extract_second_level_nodes(original_tree)

        if not blocks:
            print(f"项目 {item_id}: 未找到文本块，保留原始树")
            new_item['trees'] = original_tree
            return new_item

        # 执行自底向上聚类
        upper_structure = bottom_up_clustering(
            second_level_nodes,
            blocks,
            n_levels=4,
            threshold=0.8
        )

        # 合并树结构
        new_tree = merge_tree_structures(original_tree, upper_structure)
        new_item['trees'] = new_tree

        print(f"项目 {item_id} 处理完成")
        return new_item

    except Exception as e:
        print(f"处理项目时出错: {str(e)}")
        # 返回原始项目，确保不丢失数据
        return item


def rebuild_tree_structure(input_file_path: str, output_file_path: str, num_items=None):
    """重建树结构的主函数"""
    start_time = time.time()
    print(f"开始处理文件: {input_file_path}")

    try:
        # 读取输入文件
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)[0:100]

        if num_items is not None:
            data_list = data_list[:num_items]
            print(f"将处理前 {num_items} 个项目，共 {len(data_list)} 个")
        else:
            print(f"将处理所有 {len(data_list)} 个项目")

        if not isinstance(data_list, list):
            data_list = [data_list]
            print("警告: 输入数据不是列表，已转换为单元素列表")

        # 设置进程数
        num_processes = min(8, multiprocessing.cpu_count())
        print(f"使用 {num_processes} 个进程并行处理")

        # 使用进程池处理
        with Pool(processes=num_processes) as pool:
            processed_data = list(tqdm(
                pool.imap_unordered(process_single_item, data_list),
                total=len(data_list),
                desc="处理树结构"
            ))

        # 写入结果
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"处理完成！结果已保存到: {output_file_path}")
        print(f"总运行时间: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")

    except Exception as e:
        print(f"重建树结构时出错: {str(e)}")
        raise

    finally:
        # 确保所有进程都已清理
        if multiprocessing.current_process().name == 'MainProcess':
            for child in multiprocessing.active_children():
                child.terminate()


if __name__ == "__main__":
    try:
        start_time = time.time()

        # 设置进程启动方式
        multiprocessing.set_start_method('spawn', force=True)

        # 设置CUDA设备
        os.environ['CUDA_VISIBLE_DEVICES'] = '5'

        # 测试chat函数
        print("测试 chat API 连接...")
        test_response = chat("简单总结下什么是大语言模型")
        print(f"API 测试响应: {test_response[:100]}..." if test_response else "API 测试失败，未收到响应")

        input_file = ""
        output_file = ""

        rebuild_tree_structure(
            input_file,
            output_file,
            num_items=100  # 限制处理项目数量
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"总运行时间: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")

    except Exception as e:
        print(f"主程序执行错误: {str(e)}")

    finally:
        # 清理资源
        if multiprocessing.current_process().name == 'MainProcess':
            multiprocessing.active_children()  # 等待所有子进程结束