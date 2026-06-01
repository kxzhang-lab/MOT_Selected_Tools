from typing import Dict, Any
import json

def parse_response(response: str) -> Dict[str, Any]:
        """
        解析模型响应，提取中文描述、英文描述和 JSON 字段
        
        Args:
            response: 模型生成的原始响应文本
            
        Returns:
            解析后的字典，包含 chinese, english, json_fields
        """
        result = {
            "chinese": "",
            "english": "",
            "json_fields": {},
            "raw_response": response
        }
        
        # 尝试从响应中提取各部分
        lines = response.split('\n')
        
        current_section = None
        chinese_lines = []
        english_lines = []
        json_lines = []
        
        in_json = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # 检测章节标题
            if 'Chinese' in line_stripped or '中文' in line_stripped:
                current_section = 'chinese'
                continue
            elif 'English' in line_stripped:
                current_section = 'english'
                continue
            elif 'JSON' in line_stripped or 'json' in line_stripped:
                current_section = 'json'
                in_json = False
                continue
            
            # 收集内容
            if current_section == 'chinese':
                if line_stripped and not line_stripped.startswith('```'):
                    chinese_lines.append(line)
            elif current_section == 'english':
                if line_stripped and not line_stripped.startswith('```'):
                    english_lines.append(line)
            elif current_section == 'json':
                # 检测 JSON 代码块边界
                if line_stripped.startswith('```json'):
                    in_json = True
                    continue
                elif line_stripped.startswith('```'):
                    in_json = False
                    continue
                elif in_json and line_stripped.startswith('{'):
                    json_lines.append(line)
        
        # 组合结果
        if chinese_lines:
            result["chinese"] = '\n'.join(chinese_lines).strip()
        if english_lines:
            result["english"] = '\n'.join(english_lines).strip()
        
        # 尝试解析 JSON
        if json_lines:
            json_str = '\n'.join(json_lines).strip()
            try:
                result["json_fields"] = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    json_str = json_str.replace('```json', '').replace('```', '').strip()
                    result["json_fields"] = json.loads(json_str)
                except json.JSONDecodeError:
                    result["json_fields"] = {"parse_error": json_str[:200]}
        return result