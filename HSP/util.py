import re
import tiktoken
from typing import List, Dict, Any, Tuple

# chunk_size = 4K 可自行更改
def split_text_into_token_blocks(text: str, block_size: int = 4096,
                                 delimiters: List[str] = ['。', '！', '？', '!', '?', '.', '#', '-'],
                                 encoding_name: str = 'cl100k_base') -> List[str]:

    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    total_tokens = len(tokens)

    delimiter_token_ids = set()
    for delim in delimiters:
        delim_tokens = encoding.encode(delim)
        if len(delim_tokens) == 1:
            delimiter_token_ids.add(delim_tokens[0])
        else:
            raise ValueError(f"Delimiter '{delim}' is not a single token when encoded.")

    blocks = []
    current_position = 0

    while current_position < total_tokens:
        end_position = current_position + block_size
        if end_position >= total_tokens:
            block_tokens = tokens[current_position:]
            block_text = encoding.decode(block_tokens).strip()
            blocks.append(block_text)
            break

        current_block_tokens = tokens[current_position:end_position]

        if current_block_tokens[-1] in delimiter_token_ids:
            block_text = encoding.decode(current_block_tokens).strip()
            blocks.append(block_text)
            current_position = end_position
        else:
            split_pos = -1
            for i in range(len(current_block_tokens) - 1, -1, -1):
                if current_block_tokens[i] in delimiter_token_ids:
                    split_pos = i
                    break

            if split_pos != -1:
                split_token_end = current_position + split_pos + 1
                block_tokens = tokens[current_position:split_token_end]
                block_text = encoding.decode(block_tokens).strip()
                blocks.append(block_text)
                current_position = split_token_end
            else:
                block_text = encoding.decode(current_block_tokens).strip()
                blocks.append(block_text)
                current_position = end_position

    return blocks


def merge_node_data(existing_node: Dict[str, Any], new_node: Dict[str, Any]) -> None:
    print(f"正在合并标题为 '{new_node['title']}' 的节点")

    original_keywords = set(existing_node["keywords"])
    existing_node["keywords"] = list(set(existing_node["keywords"] + new_node["keywords"]))

    added_keywords = set(existing_node["keywords"]) - original_keywords
    if added_keywords:
        print(f"  添加了新的关键词: {', '.join(added_keywords)}")

    if new_node["summary"] and new_node["summary"] != existing_node["summary"]:
        print(f"  合并摘要信息")
        if existing_node["summary"]:
            existing_node["summary"] += " " + new_node["summary"]
        else:
            existing_node["summary"] = new_node["summary"]

    for child in new_node["children"]:
        child_title = child["title"]
        existing_children = {c["title"]: c for c in existing_node["children"]}

        if child_title in existing_children:
            print(f"  发现相同标题的子节点: '{child_title}'")
            merge_node_data(existing_children[child_title], child)
        else:
            print(f"  添加新的子节点: '{child_title}'")
            existing_node["children"].append(child)


def parse_response_to_tree(response_text: str, input_block: str) -> List[Dict[str, Any]]:
    def find_sections(text: str) -> List[tuple]:
        sections = []

        # 1) 主标题匹配 —— 只匹配文本的第一行，且位于 *** 和 *** 之间的内容
        main_pattern = re.compile(r'^\s*\*\*\*([^\*]+?)\*\*\*', re.MULTILINE)
        main_match = main_pattern.match(text.strip())  # match() 确保匹配的是第一行

        if main_match:
            main_title = main_match.group(1).strip()  # 拿到主标题文本
            main_start = 0
            main_end = main_match.end()
            sections.append((0, [], main_title, main_start, main_end))
        else:
            print("警告：未找到符合格式的主标题")
            return []  # 没有主标题直接返回

        # Find all numbered sections with updated pattern
        section_pattern = re.compile(
            r'\*\*\*'  # 开始的 ***
            r'\s*'  # 可能的空白字符
            r'(?:(\d+(?:\.\d+)*)'  # 可选的数字部分（例如 1, 1.1, 1.2.3）
            r'[\s\.]+)?'  # 数字后的空格或点号（如果有数字的话）
            r'([^*]+?)'  # 标题文本（非贪婪匹配到下一个 *）
            r'\*\*\*'  # 结束的 ***
        )

        for match in section_pattern.finditer(text):
            # 获取编号（如果存在）
            numbering_str = match.group(1) if match.group(1) else ""
            # 获取标题
            title = match.group(2).strip()

            # 如果有编号，转换为数字列表
            if numbering_str:
                numbering = [int(n) for n in numbering_str.rstrip('.').split('.')]
                level = len(numbering)
            else:
                # 如果没有编号（比如主标题），使用空列表
                numbering = []
                level = 0

            # 获取位置
            start_pos = match.start()
            sections.append((level, numbering, title, start_pos, -1))

        # Set end positions for each section
        for i in range(len(sections) - 1):
            sections[i] = (*sections[i][:4], sections[i + 1][3])
        if sections:
            sections[-1] = (*sections[-1][:4], len(text))

        # Debug output
        print("\n找到的节段：")
        for level, numbering, title, start, end in sections:
            prefix = "  " * level
            number_str = '.'.join(map(str, numbering)) if numbering else "Root"
            print(f"{prefix}{number_str}: {title}")

        return sections

    def parse_section_content(content: str) -> Dict[str, Any]:
        """Parse section content to extract keywords, summary and original text"""
        # Extract Keywords
        keywords_match = re.search(r'\*\*Keywords\*\*:\s*(.*?)\s*\*\*Summary\*\*', content, flags=re.DOTALL)
        keywords_str = keywords_match.group(1).strip() if keywords_match else ""

        # 根据分号或中文分号切分
        keywords = [kw.strip() for kw in re.split(r'[;；]', keywords_str) if kw.strip()]

        # 2) 匹配 summary
        summary_match = re.search(r'\*\*Summary\*\*:\s*(.*?)(?=\n|\Z)', content, flags=re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else ""

        # 3) 匹配 original_text
        original_text_match = re.search(r'\*\*Original Text\*\*:\s*(.*)', content, flags=re.DOTALL)
        original_text = original_text_match.group(1).strip() if original_text_match else ""

        return {
            "keywords": keywords,
            "summary": summary,
            "original_text": original_text
        }

    try:
        # Clean input text
        response_text = response_text.replace('\r\n', '\n').strip()

        print("\n=== 开始解析文本 ===")
        print("原始文本前200字符：", response_text[:200])

        # Find all sections
        sections = find_sections(response_text)
        if not sections:
            print("错误：未找到任何节段")
            return []

        print(f"\n找到的标题数量：{len(sections)}")
        for level, numbering, title, start, end in sections:
            print(f"- {'  ' * level}[{'.'.join(map(str, numbering)) if numbering else 'Root'}] {title}")

        # Create nodes map for building tree
        nodes_map = {}

        # Process each section
        for level, numbering, title, start, end in sections:
            # Parse section content
            content = response_text[start:end].strip()
            section_data = parse_section_content(content)

            # Create node dictionary
            node = {
                "title": title,
                "keywords": section_data["keywords"] if section_data["keywords"] else "",
                "summary": section_data["summary"] if section_data["summary"] else "",
                "original_text": section_data["original_text"] if section_data["original_text"] else "",
                "children": []
            }

            # Add to nodes map
            numbering_key = tuple(numbering)
            nodes_map[numbering_key] = node

            # Add to parent's children if not root
            if level > 0:
                parent_numbering = tuple(numbering[:-1])
                if parent_numbering in nodes_map:
                    parent = nodes_map[parent_numbering]
                    parent["children"].append(node)
                else:
                    print(f"警告：找不到父节点 {'.'.join(map(str, numbering[:-1]))}")

        # Get root node
        root_node = nodes_map.get(tuple(), None)
        if not root_node:
            print("错误：未找到根节点")
            return []

        # Print tree structure for verification
        print("\n=== 解析结果验证 ===")

        def print_tree(node, level=0):
            print(f"{'  ' * level}- {node['title']}")
            for child in node['children']:
                print_tree(child, level + 1)

        print_tree(root_node)

        return [root_node]

    except Exception as e:
        print(f"解析过程出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return []


def parse_tree_response(response: Dict[str, Any], idx: int) -> Tuple[List[Dict[str, Any]], int]:
    generated_response = response.get('generated_response', '')
    input_block = response.get('input_block', '')

    if not generated_response:
        return None, idx

    tree = parse_response_to_tree(generated_response, input_block)

    # 添加递归函数为每个节点添加chunk_id
    def add_chunk_id_recursive(node, chunk_id):
        node['chunk_id'] = chunk_id
        for child in node.get('children', []):
            add_chunk_id_recursive(child, chunk_id)

    # 为树中的每个节点添加chunk_id
    if tree:
        for node in tree:
            add_chunk_id_recursive(node, idx)

    return tree, idx


def merge_trees(trees: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trees:
        return {
            "title": "Merged Document",
            "keywords": [],
            "summary": "No documents to merge",
            "children": []
        }

    merged_root = {
        "title": "Merged Document",
        "keywords": [],
        "summary": "Combined results of all documents",
        "children": []
    }

    title_to_node = {}
    print('Processing trees...')

    for tree in trees:
        if not isinstance(tree, dict):
            print(f"Warning: Invalid tree type: {type(tree)}")
            continue

        root_title = tree.get("title")
        if not root_title:
            print(f"Warning: Tree missing title: {tree}")
            continue

        if root_title in title_to_node:
            print(f"\n发现重复的标题节点: '{root_title}'，开始合并处理")
            merge_node_data(title_to_node[root_title], tree)
        else:
            print(f"\n添加新的根节点: '{root_title}'")
            new_node = {
                "title": root_title,
                "keywords": tree.get("keywords", []),
                "summary": tree.get("summary", ""),
                "children": tree.get("children", []),
                "chunk_id": tree.get("chunk_id"),
            }

            if "input_block" in tree:
                new_node["original_block"] = tree["input_block"]

            title_to_node[root_title] = new_node
            merged_root["children"].append(new_node)

    print("\n树结构合并完成")
    return merged_root
