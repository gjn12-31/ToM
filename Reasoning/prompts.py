# prompts.py
"""
提供所有用于树形MapReduce推理的prompt模板
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)  # 确保 logger 被正确引用

MAP_PROMPT = """
You are provided with a portion of an article, previous reasoning result and a question. Read the article portion with previous reasoning result and follow my instructions to process it.\n\n

Question:\n{question}\n\n

The article includes four sections: title, keywords, summary, and original text. Please use the original text as the primary basis for reasoning if the original text is not null, and utilize the other information as supplementary reasoning references to attempt answering the question.\n\n

Article:\nThe article begins as follows:\n{content}\nThe article concludes here.\n\n
The output of the previous step of reasoning:\n{child_result}\n\n

Instructions:\n\n
Please extract information from the provided passage to try and answer the given question. 
Note that you only have a part of the entire text, so the information you obtain might not fully answer the question. Your task is to extract all the relevant information from this section of the article. It should cover all potentially relevant details, including at least 10 pieces of information.
Then, provide your rationale for using the extracted information to answer the question and include a confidence score. 

 Follow these steps:\n\n
 1. Extract Relevant Information(key_info): Identify and highlight the key pieces of information from the passage(mainly from the original text) that are relevant to the given question.This section should gather as much  information as possible, whether from the current content or from previous reasoning results.Note: The richer the key information, the more it contributes to reasoning out the correct answer.including at least 10 pieces of information\n
 2. Provide a Rationale(reasoning_process): This is a very important section. Please provide a detailed and convincing reasoning process. Analyze the extracted information and explain how it can be used to address the question. If the information is incomplete, discuss any assumptions or inferences you need to make.\n
 3. Answer the Question: Based on your rationale, provide the best possible answer to the question without any explanation.\nThe answer in less than 5 words.(You must give an concrete and brief answer.")\n\n
 4. Assign a Confidence Score: Assign a confidence score (out of 10) to your answer based on the completeness and reliability of the extracted information and your rationale process.If you have a high level of confidence in the answer provided above, please assign a higher score; if you believe the current answer lacks sufficient convincing evidence, please assign a lower score.\n
 
 The following is some assigning scoring cases: 
<Text: [Jerry is 18 years old this year. He can swim and wants to be an athlete.].
 assigning scoring: [Jerry can swim, 10 points; Jerry will become an athlete in the future, 7 points; Jerry will become a swimming athlete in the future, 6 points;Jerry is strong,6 points; Jerry can play chess, 0 points;Jerry likes talking,0 points]>. 
 
 Question:\n{question}\n\n

 Please strictly follow this format:
***start output***
 **key_info**: ;
 **reasoning_process**: ;
 **answer**: ;
 **confidence_score**: ;
***end output***
"""

REDUCE_PROMPT = """
You need to process a task with a very long context . 
The  way to handle this is by processing the long context chunk by chunk. 
You are provided with a question and some information extracted from each chunk. 
Each piece of information contains Extracted Information(key_info), Rationale(reasoning_process), Answer, and a Confidence Score. \n\n

Read the information and follow my instructions to process it.\n\n 
Question:\n{question}\n\n
Output information from former inference:\n{results}\n\n

Each chunk provides extracted information related to the same question, but due to partial data, conclusions from each chunk might vary. Your role is to integrate and reason through this information, weighing confidence scores to resolve any inconsistencies. Please follow steps as:
1. You should gather as much extracted information as possible from each chunk which you think is relevant to the question. Information from each chunk might not fully answer the question, your task is to gather them so that the next reasoning will be better. The extracted information includes at least 20 pieces of information.\n\n
2. Provide a complete and detailed rationale based on the question and the currently extracted information.\n\n
3. Then provide the final answer(Less than 6 words.You must give an concrete and brief answer.").\n\n
4. Assign a confidence score (out of 10) to your answer based on the completeness and reliability of the extracted information and your rationale process

The following is some assigning scoring cases: 
<Text: [Jerry is 18 years old this year. He can swim and wants to be an athlete.].
 assigning scoring: [Jerry can swim, 10 points; Jerry will become an athlete in the future, 7 points; Jerry will become a swimming athlete in the future, 6 points;Jerry is strong,6 points; Jerry can play chess, 0 points;Jerry likes talking,0 points]>. 

Question:\n{question}\n\n

Please strictly follow this format:
***start output***
 **key_info**: ;
 **reasoning_process**: ;
 **answer**: ;
 **confidence_score**: ;
***end output***
"""

def get_prompts() -> dict:
    """获取所有prompts"""
    return {
        'map_prompt': MAP_PROMPT,
        'reduce_prompt': REDUCE_PROMPT
    }

# 如果需要从配置文件加载prompt
def load_prompts_from_config(config_path: str) -> dict:
    """从配置文件加载prompt模板"""
    import json
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return {
        'map_prompt': config.get('map_prompt', MAP_PROMPT),
        'reduce_prompt': config.get('reduce_prompt', REDUCE_PROMPT)
    }

# 格式化prompt
def format_map_prompt(question: str, content: str, child_result: Optional[str] = None) -> str:
    child_result_str = f"子节点推理结果：\n{child_result}" if child_result else ""
    prompt = MAP_PROMPT.format(question=question, content=content, child_result=child_result_str)
    logger.debug(f"Formatted map prompt: {prompt}")  # 添加调试日志
    return prompt

def format_reduce_prompt(question: str, results: str) -> str:
    prompt = REDUCE_PROMPT.format(question=question, results=results)
    logger.debug(f"Formatted reduce prompt: {prompt}")  # 添加调试日志
    return prompt
