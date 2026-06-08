import json5
import re

def parse_with_json5(raw_response):
    """使用 json5 解析模型响应中的 JSON 内容"""
    result = {"json_content": None, "raw_response": raw_response}
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', raw_response)
    if not json_match:
        raise ValueError("未找到 JSON 代码块")
    json_str = json_match.group(1).strip()
    result["json_content"] = json5.loads(json_str)
    return result