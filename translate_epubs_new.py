import os
import glob
import json
import zipfile
import sys
import typer
import re
import opencc
from lxml import etree
from bs4 import BeautifulSoup
import logging
import subprocess
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')

class TwoPassNovelTranslator:
    """
    A professional-grade EPUB translator designed for Japanese light novels.
    Optimized for API backends (mlx_lm.server, vLLM, llama.cpp) to leverage KV Prefix Caching.
    
    Features:
    - Fixed glossary tables (glossary.txt)
    - 2-pass architecture (Lore scanning -> Contextual translation)
    - Batch/Chunked translations for high-speed throughput
    - Sliding window for massive chapters (50k+ chars)
    - Sliding Context Window (History) for consistent narrative flow
    - <ruby> character handling
    - Configurable Multiple Quality Assurance (QA) passes
    """
    
    def __init__(self, base_url="http://localhost:8080/v1", api_key="not-needed", model_name="default", dict_path=None, max_tokens=8192, verbose=False, attempts=2, history=25, presence_penalty=0.0, to_traditional=True, temperature=1.0, chapter_abbrev=False):
        self.cc = opencc.OpenCC('s2hk.json')
        self.cc_back = opencc.OpenCC('t2s.json')
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.attempts = attempts
        self.history = history
        self.presence_penalty = presence_penalty
        self.to_traditional = to_traditional
        self.model_name = model_name
        self.temperature = temperature
        self.chapter_abbrev = chapter_abbrev 
        
        print(f"[*] Connecting to OpenAI-compatible server at {base_url}")
        try:
            from openai import OpenAI
        except ImportError:
            print("[!] Error: The 'openai' package is required.")
            print("[!] Please install it by running: pip install openai")
            sys.exit(1)
            
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        
        # Test connection
        try:
            self.client.models.list()
            print("[+] Successfully connected to the inference server.")
        except Exception as e:
            print(f"[-] Warning: Could not fetch model list from server (Error: {e}). Proceeding anyway...")
        
        self.global_glossary = {}
        self.predefined_dict = self._load_dictionary(dict_path)

    def _remove_think_tags(self, text):
        if '</think>' in text:
            r = text.split('</think>', 1)[-1]
        elif '<channel|>' in text:
            r = text.split('<channel|>', 1)[-1]
        elif '<|channel|>thought' in text:  # The Gemma fail-safe
            r = text.split('<|channel|>thought', 1)[-1]
        else:
            r = text
            
        return r.strip()

    def _finalize_text(self, text):
        """Finalize translated text with smart quotes replacement and CC conversion."""
        if not text:
            return text
            
        # 1. Replace smart quotes
        text = text.replace('“', '「').replace('”', '」')
        
        # 2. Replace straight double quotes alternately
        parts = text.split('"')
        if len(parts) > 1:
            new_text = parts[0]
            for i in range(1, len(parts)):
                if i % 2 != 0:
                    new_text += '「' + parts[i]
                else:
                    new_text += '」' + parts[i]
            text = new_text
            
        # 3. Chinese Conversion
        if self.to_traditional:
            text = self.cc.convert(text)
        else:
            text = self.cc_back.convert(text)
            
        return text

    def _load_dictionary(self, dict_path):
        user_dict = {}
        if dict_path and os.path.exists(dict_path):
            print(f"[*] Loading predefined dictionary from: {dict_path}")
            with open(dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.split('#')[0].strip()
                    if '=>' in line:
                        parts = line.split('=>', 1)
                        jp_name = parts[0].strip()
                        zh_name = parts[1].strip()
                        if jp_name and zh_name:
                            # Keep glossary in Simplified Chinese
                            user_dict[jp_name] = zh_name
            print(f"[+] Loaded {len(user_dict)} fixed translations.")
        elif dict_path:
            print(f"[-] Predefined dictionary '{dict_path}' not found. Proceeding without it.")
        return user_dict

    def _ask_llm(self, system_prompt, user_prompt, max_tokens=None, is_json=False):
        max_tokens = max_tokens or self.max_tokens
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
                presence_penalty=0.0 if is_json else self.presence_penalty,
                top_p=0.95,
                frequency_penalty=0.05,
                extra_body={
                    "min_p": 0.05
                }
            )
            
            content = response.choices[0].message.content
            if content is None:
                return ""
                
            if self.verbose:
                print(f"\n[LLM Response]:\n{content}\n")
                
            return self._remove_think_tags(content)
            
        except Exception as e:
            print(f"    [!] API Communication Error: {e}")
            return ""

    def _extract_json(self, text):
        fence = "`" + "`" + "`"
        start_marker = fence + "json"
        start_idx = text.find(start_marker)
        
        if start_idx == -1:
            start_idx = text.find(fence)
            if start_idx != -1:
                start_idx += len(fence)
        else:
            start_idx += len(start_marker)
            
        end_idx = text.rfind(fence)
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx:end_idx].strip()
        else:
            first_brace, first_bracket = text.find('{'), text.find('[')
            last_brace, last_bracket = text.rfind('}'), text.rfind(']')
            starts = [i for i in (first_brace, first_bracket) if i != -1]
            ends = [i for i in (last_brace, last_bracket) if i != -1]
            if starts and ends:
                json_str = text[min(starts):max(ends)+1]
            else:
                raise ValueError("No JSON boundaries found.")
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError("Failed to decode JSON.")

    def _ask_llm_json(self, system_prompt, user_prompt, max_retries=3, max_tokens=None):
        for attempt in range(max_retries):
            response = self._ask_llm(system_prompt, user_prompt, max_tokens=max_tokens, is_json=True)
            try:
                return self._extract_json(response)
            except ValueError as e:
                print(f"    [!] JSON Error: {e}. Retrying ({attempt + 1}/{max_retries})...")
                if attempt == max_retries - 1: 
                    print(f"      Still receiving non-JSON output as followed: {response}")
                    return {}

    def _extract_text_with_ruby(self, tag):
        tag_html = etree.tostring(tag, encoding='unicode', method='html')
        tag_copy = BeautifulSoup(tag_html, 'html.parser')
        
        for ruby in tag_copy.find_all('ruby'):
            rt_text = "".join([rt.get_text() for rt in ruby.find_all('rt')])
            for t in ruby.find_all(['rt', 'rp']): 
                t.decompose()
            base_text = ruby.get_text().strip()
            ruby.replace_with(f"{base_text}({rt_text})" if rt_text else base_text)
            
        return tag_copy.get_text().strip()

    def _has_japanese(self, text):
        return bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text))

    def _parse_xml(self, content):
        if isinstance(content, str):
            content = content.encode('utf-8')
        parser = etree.XMLParser(recover=True, resolve_entities=False)
        tree = etree.fromstring(content, parser=parser)
        return tree

    def _serialize_xml(self, tree):
        new_tree = tree.getroottree()
        doctype = new_tree.docinfo.doctype
        if doctype:
            return etree.tostring(new_tree, encoding='utf-8', xml_declaration=True, doctype=doctype, method='xml')
        return etree.tostring(new_tree, encoding='utf-8', xml_declaration=True, method='xml')

    def _get_epub_metadata(self, zin):
        try:
            container_xml = zin.read('META-INF/container.xml')
            container_tree = self._parse_xml(container_xml)
            opf_path = container_tree.xpath('//*[local-name()="rootfile"]/@full-path')[0]
            
            opf_content = zin.read(opf_path)
            opf_tree = self._parse_xml(opf_content)
            
            manifest = {}
            nav_paths = []
            
            for item in opf_tree.xpath('//*[local-name()="manifest"]/*[local-name()="item"]'):
                item_id = item.get('id')
                href = item.get('href')
                manifest[item_id] = href
                
                if item.get('properties') == 'nav' or item.get('media-type') == 'application/x-dtbncx+xml':
                    nav_paths.append(href)
                    
            spine = []
            for itemref in opf_tree.xpath('//*[local-name()="spine"]/*[local-name()="itemref"]'):
                spine.append(itemref.get('idref'))
                
            opf_base = os.path.dirname(opf_path)
            
            def resolve_path(href):
                if opf_base:
                    return os.path.normpath(os.path.join(opf_base, href)).replace('\\', '/')
                return href
                
            spine_paths = []
            for idref in spine:
                if idref in manifest:
                    spine_paths.append(resolve_path(manifest[idref]))
                    
            toc_paths = [resolve_path(p) for p in nav_paths]
            
            return spine_paths, toc_paths
        except Exception as e:
            logging.error(f"Failed to parse EPUB metadata: {e}")
            return [], []

    # ==========================================
    # PHASE 1: GLOSSARY BUILDING (Sliding Window)
    # ==========================================
    
    def scan_for_entities(self, text_chunk):
        dict_context = f"参考译名表: {json.dumps(self.predefined_dict, ensure_ascii=False)}\n" if self.predefined_dict else ""
        system_prompt = (
            "你是一个轻小说专家和设定集管理员。\n"
            "任务：从日文轻小说的段落中提取非汉字名字及片假名专有名词，包括人名、地名、特别活动名、特别物品名等，翻译成中文，制作成翻译用的术语表。\n"
            "步骤：\n"
            "1. 从提供的日文文本片段中提取所有非汉字名字及片假名专有名词，作为日文原名（复合词和汉字名称除外）。\n"
            "2. 人人物名必须除去称谓（如「様」等）。\n"
            "3. 如果中文译名和日文原文相同请不要提取。\n"
            "4. 「现有实体表」里面已经有的，不要再提取，除非有未知的资料可以补充。如果「参考译名表」里面有，但「现有实体表」没有的，则可以提取。两者都没有的则要提取。\n"
            "5. 如果是一般日语词典里面会有的词语，请不要提取。\n"
            "6. 过滤不需要的词语后，如果在「参考译名表」里面已经有，请使用该译名，没有则请翻译一个优雅的中文译名，所有输出必须为中文。\n"
            "7. 有的片假名（尤其是四个字的）可能是其他复合名词的略写，如有遇到，请提取翻译。\n"
            "8. 通过语境（如自称、他称、描述）推断性别和头衔/称谓。如果没有信息可以推断，或者不是人物，性别填未知，不是人物类别填类别。\n"
            "9. 严格遵守 Output Format Sample 输出合法的 JSON 格式，以日文原名为 key，value 必须为 Object，包含指定的字段。不要输出任何解释性文字和 Output Format Sample 没有定义的 JSON 格式以外的内容。\n"
            "10. 如果没有找到新实体，请返回空 JSON {}。\n\n"
            f"【现有实体表】: {json.dumps(self.global_glossary, ensure_ascii=False)}\n"
            f"【{dict_context}】"
        )
        user_prompt = (
            f"日文文本片段:\n{text_chunk}\n\n"
            "Output Format Sample: {\"日文原名\": {\"zh_name\": \"中文名\", \"type\": \"人名/地名/物品/活动/其他/未知\", \"gender\": \"男/女/未知\"}}\n"
        )
        updates = self._ask_llm_json(system_prompt, user_prompt)
        for name, data in updates.items():
            if isinstance(data, dict):
                zh_name_val = data.get('zh_name', '')
                # Smart Filter: Skip if exact same (mostly pure Kanji that translates cleanly anyway)
                if name == zh_name_val or name == self.cc.convert(zh_name_val):
                    continue
                
                if name in self.predefined_dict: 
                    data['zh_name'] = self.predefined_dict[name]
                if name not in self.global_glossary:
                    print(f"    [New Entity] {name} -> {data.get('zh_name')}")
                self.global_glossary[name] = data

    def run_lore_pass(self, epub_files):
        print("\n>>> STARTING PASS 1: LORE SCANNING (Building Master Glossary)")
        for filepath in epub_files:
            print(f"[*] Scanning {os.path.basename(filepath)}...")
            with zipfile.ZipFile(filepath, 'r') as zin:
                spine_paths, _ = self._get_epub_metadata(zin)
                
                for doc_path in spine_paths:
                    print(f"\nProcessing {doc_path}...")
                    
                    try:
                        content = zin.read(doc_path)
                        tree = self._parse_xml(content)
                    except Exception as e:
                        print(f"    [!] XML Parse error, skipping: {e}")
                        continue
                        
                    tags = tree.xpath('//*[local-name()="p" or local-name()="h1" or local-name()="h2" or local-name()="h3" or local-name()="h4"]')
                    full_text = " ".join([self._extract_text_with_ruby(t) for t in tags])
                    if len(full_text) < 100: continue
                    
                    window_size = 5000
                    overlap = 500
                    for start in range(0, len(full_text), window_size - overlap):
                        chunk = full_text[start:start + window_size]
                        self.scan_for_entities(chunk)
                        if len(chunk) < window_size: 
                            break
            
        for jp_name, zh_name in self.predefined_dict.items():
            if jp_name not in self.global_glossary:
                self.global_glossary[jp_name] = {
                    "zh_name": zh_name,
                    "gender": "未知",
                    "type": ""
                }
                print(f"    [Forced Entity] {jp_name} -> {zh_name} (from predefined dictionary)")

        with open("final_glossary.json", "w", encoding="utf-8") as f:
            json.dump(self.global_glossary, f, ensure_ascii=False, indent=2)

    # ==========================================
    # PHASE 2: TRANSLATION (Headers & TOC)
    # ==========================================

    def resolve_contextual_names(self, chapter_text):
        system_prompt = (
            "你是一个逻辑推理助手。\n"
            "任务：找出【文本】中出现的日文昵称，或常见的片假名缩写，并将其映射到正确的含义。\n"
            "【映射规则】：\n"
            "1. 从提供的日文文本片段中找出人物的日文昵称（如「あーちゃん」），或常见的片假名缩写（如「ギルマス」）。\n"
            "2. 如果是常见的片假名缩写，请直接输出中文翻译到映射表。\n"
            "3. 如果是日文昵称，请根据【术语表】与上下文，找出该人物的「完整日文原名」。如果对应不到则不要输出。\n"
            "请输出 JSON 格式映射表：{\"缩写/昵称\": \"对应的完整日文原名\"}\n"
            "如果没有发现任何缩写或昵称，请务必回传空对象 {}。\n\n"
            f"【术语表】: {json.dumps(self.global_glossary, ensure_ascii=False)}\n"
        )
        user_prompt = (
            f"本章文本片段:\n{chapter_text[:2500]}\n\n"
            "JSON Mappings:"
        )
        return self._ask_llm_json(system_prompt, user_prompt)

    def translate_single_line(self, jp_text, chapter_abbrevs, history_context=""):
        system_prompt = (
            "你是一位顶尖的轻小说翻译专家，能严格遵守以下要求将提供的日文翻译为轻小说风格的中文。\n"
            "要求：\n"
            "1. 如有提供作参考的前文，请先阅读及理解，在翻译时保持逻辑正确。\n"
            "2. 翻译前必须先理解内容，并找出日文里有在术语表中出现的词汇。\n"
            "3. 小心分析原文的句形，正确判断是主动形或受身形等，确保没有混淆。遇到受身形等句式省略主语时，请综合考虑上下文小心思考，如果清楚明白，请补全正确的主语或宾语。如不清楚，请避免补上不确定的主语或宾语。\n"
            "4. 严格遵守提供的【术语表】。遇到术语表中的词汇，必须使用对应的中文翻译。\n"
            "5. 把日文翻译成通顺的中文，尽量保留句子原来的语气和意思。\n"
            "6. 【重要】不允许改变原文中的方引号（「」）。不准添加或使用全形或半形的西式引号。除此以外，请使用正确的全形标点符号。\n"
            "7. 【超级重要】翻译结果中不准使用英语，除了原文里面的英语专有名称以外，必须把「such」改为「这种」。\n"
            "8. 翻译成中文后再和日文原文校对一次，确保原文的意思正确地表达。\n"
            "9. 【极度重要】直接输出纯中文翻译，绝对不要包含任何解释或 Markdown 标签。\n\n"
            f"【全局术语表】: {json.dumps(self.global_glossary, ensure_ascii=False)}\n"
            f"【本章简称映射表】: {json.dumps(chapter_abbrevs, ensure_ascii=False)}\n"
            "请勿输出未经翻译的日文原文。\n"
        )
        
        context_block = f"历史翻译上下文 (仅供参考, 请勿重新翻译):\n{history_context}\n\n" if history_context else ""
            
        user_prompt = (
            f"{context_block}"
            f"待翻译日文原文: {jp_text}\n\n"
            "翻译结果:"
        )
        res = self._ask_llm(system_prompt, user_prompt, max_tokens=self.max_tokens)
        return res.strip()

    def translate_chunk(self, jp_texts, chapter_abbrevs, history_context=""):
        system_prompt = (
            "你是一位顶尖的轻小说翻译专家，能严格遵守以下要求将提供的多个日文段落逐一翻译为轻小说风格的中文。\n"
            "要求：\n"
            "1. 如有提供作参考的前文，请先阅读及理解，在翻译时保持逻辑正确。\n"
            "2. 翻译前必须先理解内容，并找出每段日文里有在术语表中出现的词汇。\n"
            "3. 小心分析原文的句形，正确判断是主动形或受身形等，确保没有混淆。遇到受身形等句式省略主语时，请综合考虑上下文小心思考，如果清楚明白，请补全正确的主语或宾语。如不清楚，请避免补上不确定的主语或宾语。\n"
            "4. 严格遵守提供的【术语表】。遇到术语表中的词汇，必须使用对应的中文翻译。\n"
            "5. 逐一把每段日文翻译成通顺的中文，尽量保留句子原来的语气和意思。\n"
            "6. 【重要】不允许改变原文中的方引号（「」）。不准添加或使用全形或半形的西式引号。除此以外，请使用正确的全形标点符号。\n"
            "7. 【超级重要】翻译结果中不准使用英语，除了原文里面的英语专有名称以外，必须把「such」改为「这种」。\n"
            "8. 翻译成中文后再和日文段落原文对比一次，确保原文的意思正确地表达。\n"
            "9. 【极度重要】不允许使用 JSON！使用指定的分隔符，输出纯文本。\n"
            "10. 返回的翻译段落数量在使用分隔符隔开后，必须与原文段落数量完全一致，绝对不要包含任何解释或 Markdown 标签。\n\n"
            f"【全局术语表】: {json.dumps(self.global_glossary, ensure_ascii=False)}\n"
            f"【本章简称映射表】: {json.dumps(chapter_abbrevs, ensure_ascii=False)}\n"
            "请勿输出未经翻译的日文原文。\n"
        )
        
        context_block = f"历史翻译上下文 (仅供参考, 请勿重新翻译):\n{history_context}\n\n" if history_context else ""
            
        for attempt in range(self.attempts):
            if attempt % 3 == 0:
                delimiter = "===="
            elif attempt % 3 == 1:
                delimiter = "$$$$"
            else:
                delimiter = "~~~~"
                
            user_prompt = (
                "请根据以下要求执行翻译任务，输出繁体中文。\n"
                "【重要】不允许改变原文中的方引号（「」）。除此以外，请使用正确的全形标点符号。\n"
                f"【分隔符规则】：为了区分不同的段落，你必须在每个翻译段落之间单独使用 `{delimiter}` 作为换行分隔符。\n"
                f"格式范例：\n第一段翻译\n{delimiter}\n第二段翻译\n\n"
                f"{context_block}"
                f"待翻译日文段落 (共 {len(jp_texts)} 段):\n{json.dumps(jp_texts, ensure_ascii=False)}\n\n"
                f"中文翻译结果 (使用 {delimiter} 分隔):"
            )

            res = self._ask_llm(system_prompt, user_prompt, max_tokens=self.max_tokens)
            res_clean = re.sub(r'^```.*?\n|```$', '', res.strip(), flags=re.MULTILINE).strip()
            
            escaped_delimiter = re.escape(delimiter)
            out = [t.strip() for t in re.split(rf'\n*\s*{escaped_delimiter}\s*\n*', res_clean)]
            
            if out and not out[-1]:
                out.pop()

            if len(out) != len(jp_texts):
                # damnit but ...
                out = [t.strip() for t in re.split(rf'\n*\s*{escaped_delimiter}\s*\n*|\n+', res_clean)]
                if out and not out[-1]:
                    out.pop()
                
            if len(out) == len(jp_texts): 
                failed_translation = False
                for orig_line, trans_line in zip(jp_texts, out):
                    # Check if a translated line is exactly the same as the original input
                    if orig_line.strip() == trans_line.strip() and self._has_japanese(trans_line):
                        failed_translation = True
                        break
                
                if failed_translation:
                    print(f"    [!] Chunk translation failed: Detected an identical unchanged line in the output. Retrying...")
                    continue
                    
                return out
            else:
                print(f"    [!] Received chunk with wrong length: expected {len(jp_texts)}, got {len(out)}. Retrying with new delimiter...")
                
        print(f"    [!] Chunk translation failed {self.attempts} times. Falling back to line-by-line translation.")
        
        fallback_translations = []
        dynamic_history = history_context.split('\n') if history_context else []
        
        for i, text in enumerate(jp_texts):
            print(f"      - Line-by-line fallback ({i+1}/{len(jp_texts)})...")
            current_history_str = "\n".join(dynamic_history[-self.history:])
            zh_text = self.translate_single_line(text, chapter_abbrevs, current_history_str)
            fallback_translations.append(zh_text)
            if zh_text:
                dynamic_history.append(zh_text)
            
        return fallback_translations

    def _translate_toc_content(self, content, chunk_size=12):
        try:
            tree = self._parse_xml(content)
        except Exception as e:
            print(f"    [!] Failed to parse TOC XML: {e}")
            return content
            
        tags = tree.xpath('//*[local-name()="a" or local-name()="span" or local-name()="h1" or local-name()="h2" or local-name()="h3" or local-name()="text"]')
        if not tags: return content
        
        valid_tags = [(tag, "".join(tag.itertext()).strip()) for tag in tags if "".join(tag.itertext()).strip()]
        for i in range(0, len(valid_tags), chunk_size):
            chunk = valid_tags[i:i+chunk_size]
            zh_batch = self.translate_chunk([t[1] for t in chunk], {})
            
            for (tag, _), zh in zip(chunk, zh_batch):
                for child in list(tag):
                    tag.remove(child)
                if zh:
                    finalized_zh = self._finalize_text(zh)
                    tag.text = finalized_zh if finalized_zh else _
                else:
                    tag.text = _
                
        return self._serialize_xml(tree)

    def run_translation_pass(self, epub_files, chunk_size=12):
        print("\n>>> STARTING PASS 2: BATCH TRANSLATION")
        translated_paths = []
        for filepath in epub_files:
            print(f"[*] Processing {os.path.basename(filepath)}...")
            modified_files = {}
            
            with zipfile.ZipFile(filepath, 'r') as zin:
                spine_paths, toc_paths = self._get_epub_metadata(zin)
                
                for doc_path in spine_paths:
                    try:
                        content = zin.read(doc_path)
                        tree = self._parse_xml(content)
                    except Exception as e:
                        print(f"    [!] Failed to parse document XML: {e}")
                        continue
                        
                    tags = tree.xpath('//*[local-name()="p" or local-name()="h1" or local-name()="h2" or local-name()="h3" or local-name()="h4"]')
                    if not tags: continue
                    
                    if self.chapter_abbrev:
                        chapter_text = " ".join([self._extract_text_with_ruby(t) for t in tags[:2000]]) 
                        chapter_abbrevs = self.resolve_contextual_names(chapter_text)
                    else:
                        chapter_abbrevs = {}
                    
                    print(f"    - Translating Document: {doc_path} ({len(tags)} blocks)")
                    
                    valid_tags = []
                    for tag in tags:
                        txt = self._extract_text_with_ruby(tag)
                        if txt: valid_tags.append((tag, txt))
                    
                    previous_translations = []
                    
                    for i in range(0, len(valid_tags), chunk_size):
                        chunk = valid_tags[i:i+chunk_size]
                        jp_texts = [t[1] for t in chunk]
                        
                        history_context = "\n".join(previous_translations) if previous_translations else ""
                        
                        zh_batch = self.translate_chunk(jp_texts, chapter_abbrevs, history_context)
                        
                        if zh_batch:
                            valid_zh = [zh for zh in zh_batch if zh]
                            if valid_zh:
                                previous_translations = valid_zh[-self.history:]
                        
                        for (tag, original_txt), zh in zip(chunk, zh_batch):
                            current_style = tag.get('style', '')
                            tag.set('style', f"{current_style}; opacity: 0.4;".strip('; '))
                            
                            new_tag = etree.Element(tag.tag)
                            if zh:
                                finalized_zh = self._finalize_text(zh)
                                new_tag.text = finalized_zh if finalized_zh else original_txt
                            else:
                                new_tag.text = original_txt
                            
                            parent = tag.getparent()
                            parent.insert(parent.index(tag) + 1, new_tag)
                    
                    modified_files[doc_path] = self._serialize_xml(tree)

                for toc_path in toc_paths:
                    print(f"    [*] Translating Table of Contents: {toc_path}")
                    try:
                        content = zin.read(toc_path)
                        modified_files[toc_path] = self._translate_toc_content(content, chunk_size)
                    except Exception as e:
                        print(f"    [!] Failed to process TOC: {e}")
            
            name, ext = os.path.splitext(filepath)
            out_path = f"{name}_zh{ext}"
            
            with zipfile.ZipFile(out_path, 'w') as zout:
                with zipfile.ZipFile(filepath, 'r') as zin:
                    if 'mimetype' in zin.namelist():
                        zout.writestr('mimetype', zin.read('mimetype'), compress_type=zipfile.ZIP_STORED)
                    for item in zin.infolist():
                        if item.filename == 'mimetype':
                            continue
                        if item.filename in modified_files:
                            zout.writestr(item.filename, modified_files[item.filename], compress_type=zipfile.ZIP_DEFLATED)
                        else:
                            zout.writestr(item.filename, zin.read(item.filename), compress_type=zipfile.ZIP_DEFLATED)
                            
            translated_paths.append(out_path)
            print(f"[+] Saved Translated Book: {out_path}")
            
        return translated_paths

    # ==========================================
    # PHASE 3: QUALITY ASSURANCE (OPTIONAL)
    # ==========================================

    def run_qa_pass(self, translated_files, chunk_size=12, pass_index=1):
        print(f"\n>>> STARTING PASS 3: QUALITY ASSURANCE (Iteration {pass_index})")
        qa_files = []
        for filepath in translated_files:
            print(f"[*] QA Checking {os.path.basename(filepath)}...")
            modified_files = {}
            
            with zipfile.ZipFile(filepath, 'r') as zin:
                spine_paths, _ = self._get_epub_metadata(zin)
                
                for doc_path in spine_paths:
                    try:
                        content = zin.read(doc_path)
                        tree = self._parse_xml(content)
                    except Exception as e:
                        print(f"    [!] Failed to parse document XML: {e}")
                        continue
                        
                    tags = tree.xpath('//*[local-name()="p" or local-name()="h1" or local-name()="h2" or local-name()="h3" or local-name()="h4"]')
                    if not tags: continue
                    
                    pairs = []
                    for i in range(len(tags) - 1):
                        style = tags[i].get('style', '')
                        if 'opacity: 0.4' in style or 'opacity:0.4' in style:
                            pairs.append({
                                'index': i + 1, 
                                'jp': "".join(tags[i].itertext()).strip(),
                                'zh': "".join(tags[i+1].itertext()).strip()
                            })
                    
                    if not pairs: continue
                    
                    for i in range(0, len(pairs), chunk_size):
                        chunk = pairs[i:i+chunk_size]
                        system_prompt = (
                            "你是一位极度严格的轻小说校对编辑。\n"
                            "任务：检查日文原文与中文翻译的一致性。\n"
                            "请阅读待校对段落的中文部分（zh），比较日语原文(jp），根据以下规则进行编辑。\n"
                            "【规则】：\n"
                            "1. 严格检查【术语表】。如果译文没有使用术语表中的规定译名，必须修改。\n"
                            "2. 修正明显的主语推断错误、代词错置或性别错误。\n"
                            "3. 【重要】如果原文是使用方引号（「」）的话，而译文改为西式引号，必须改回方引号，和原文一样。\n"
                            "4. 如果段落 jp 和 zh 没有分别，或者 zh 有很多日文文字，请重新翻译。\n"
                            "5. 如果翻译结果中有英语，除了原文里面的英语专有名称以外，请重新翻译，例如把「such」改为「这种」。\n"
                            "6. 如果译文文法不对或语句不通顺，请修改成为通顺的语句。\n"
                            "7. 如果译文已经通顺且没有违反上述各点，【绝对不要】进行修辞性或风格性的润色！不要过度修改！绝对不要包含任何解释或 Markdown 标签！\n"
                            "输出要求：若有错，回传 JSON {\"id1\": \"修正后的中文1\",\"id2\": \"修正后的中文2\"}。若完全没错，【必须】回传空对象 {}。\n\n"
                            f"【术语表】: {json.dumps(self.global_glossary, ensure_ascii=False)}\n"
                        )
                        eval_payload = [{"id": str(p['index']), "jp": p['jp'], "zh": self.cc_back.convert(p['zh'])} for p in chunk]
                        user_prompt = (
                            f"待校对段落:\n{json.dumps(eval_payload, ensure_ascii=False)}\n\n"
                            "修正 JSON:"
                        )
                        
                        corrections = self._ask_llm_json(system_prompt, user_prompt, max_tokens=self.max_tokens)
                        if corrections and isinstance(corrections,dict):
                            for idx_str, corrected_text in corrections.items():
                                try:
                                    target_tag = tags[int(idx_str)]
                                    target_tag.text = self._finalize_text(corrected_text)
                                    for child in list(target_tag):
                                        target_tag.remove(child)
                                    print(f"      [QA Fixed] Document: {doc_path}, Block {idx_str}")
                                except (ValueError, IndexError):
                                    pass

                    modified_files[doc_path] = self._serialize_xml(tree)

            if "_zh.epub" in filepath:
                name = filepath.replace("_zh.epub", f"_qa{pass_index}.epub")
            else:
                name = re.sub(r'_qa\d+\.epub$', f'_qa{pass_index}.epub', filepath)
                
            with zipfile.ZipFile(name, 'w') as zout:
                with zipfile.ZipFile(filepath, 'r') as zin:
                    if 'mimetype' in zin.namelist():
                        zout.writestr('mimetype', zin.read('mimetype'), compress_type=zipfile.ZIP_STORED)
                    for item in zin.infolist():
                        if item.filename == 'mimetype':
                            continue
                        if item.filename in modified_files:
                            zout.writestr(item.filename, modified_files[item.filename], compress_type=zipfile.ZIP_DEFLATED)
                        else:
                            zout.writestr(item.filename, zin.read(item.filename), compress_type=zipfile.ZIP_DEFLATED)
                            
            print(f"[+] Saved QA Iteration {pass_index} Book: {name}")
            qa_files.append(name)
            
        return qa_files

    def start(self, folder, qa_passes=0, chunk_size=12, qa_only=False, glossary_only=False, add_glossary=False, final_glossary=False):
        glossary_path = "final_glossary.json"
        
        # Load existing glossary for appending or QA checks
        if add_glossary or qa_only or final_glossary:
            if os.path.exists(glossary_path):
                with open(glossary_path, "r", encoding="utf-8") as f:
                    self.global_glossary = json.load(f)
                print(f"[*] Loaded existing {glossary_path} ({len(self.global_glossary)} entries)")
            else:
                if qa_only:
                    print(f"[-] {glossary_path} not found. A master glossary is required for QA passes.")
                    return
                print(f"[-] {glossary_path} not found. Starting with a fresh glossary.")

        if glossary_only:
            files = sorted(glob.glob(os.path.join(folder, "*.epub")))
            target_files = [f for f in files if not f.endswith("_zh.epub") and "_final.epub" not in f and "_qa" not in f]
            
            if not target_files:
                print(f"[-] No valid original EPUBs found in {folder} for glossary extraction.")
                return
                
            print(f"[*] Bypassing Translation & QA. Starting Glossary Extraction only on {len(target_files)} volumes...")
            self.run_lore_pass(target_files)
            print(f"[+] Finished Glossary Extraction. Master glossary saved to final_glossary.json")
            return

        if qa_only:
            if qa_passes <= 0:
                print("[*] 'qa-only' mode enabled with no --qa-pass specified. Defaulting to 1 pass.")
                qa_passes = 1
                
            translated_files = sorted(glob.glob(os.path.join(folder, "*_zh.epub")))
            if not translated_files:
                print(f"[-] No translated _zh.epub files found in {folder} to perform QA on.")
                return
                
            print(f"[*] Bypassing Lore & Translation. Starting QA directly on {len(translated_files)} translated volumes...")
            current_files = translated_files
            for i in range(1, qa_passes + 1):
                current_files = self.run_qa_pass(current_files, chunk_size=chunk_size, pass_index=i)
                
            for f in current_files:
                final_name = re.sub(r'_qa\d+\.epub$', '_final.epub', f)
                if os.path.exists(final_name):
                    os.remove(final_name)
                os.rename(f, final_name)
                print(f"[+] Finished all QA passes. Final book saved as: {final_name}")
            return

        # Normal 2-Pass + QA Flow
        files = sorted(glob.glob(os.path.join(folder, "*.epub")))
        target_files = [f for f in files if not f.endswith("_zh.epub") and "_final.epub" not in f and "_qa" not in f]
        
        if not target_files:
            print(f"[-] No valid original EPUBs found in {folder}.")
            return

        print(f"[*] Found {len(target_files)} volumes. Starting series translation...")
        if final_glossary:
            print("[*] Skipping glossary scanning.")
        else:
            self.run_lore_pass(target_files)
        translated = self.run_translation_pass(target_files, chunk_size=chunk_size)
        
        if qa_passes > 0:
            current_files = translated
            for i in range(1, qa_passes + 1):
                current_files = self.run_qa_pass(current_files, chunk_size=chunk_size, pass_index=i)
                
            for f in current_files:
                final_name = re.sub(r'_qa\d+\.epub$', '_final.epub', f)
                if os.path.exists(final_name):
                    os.remove(final_name)
                os.rename(f, final_name)
                print(f"[+] Finished all QA passes. Final book saved as: {final_name}")
        else:
            print("\n[-] QA Pass skipped by configuration.")

def run_streamlit_ui():
    """Renders the Streamlit Web UI interface."""
    try:
        import streamlit as st
    except ImportError:
        print("[!] Error: The 'streamlit' package is required to run the Web UI.")
        print("[!] Please install it by running: pip install streamlit")
        sys.exit(1)
        
    st.set_page_config(page_title="EPUB 翻译器", layout="wide")
    st.title("EPUB 日语小说翻译器")
    
    with st.form("translation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("网络及 API 配置")
            base_url = st.text_input("链接 URL", value="http://localhost:8080/v1", help="Base URL of the OpenAI-compatible server")
            api_key = st.text_input("API Key", value="not-needed", type="password", help="API key for the server")
            model = st.text_input("模型名字", value="default", help="Model name to pass to the API")
            temperature = st.number_input("Temperature", value=1.0, step=0.1)
            presence_penalty = st.number_input("Presence Penalty", value=0.0, step=0.1)
            max_tokens = st.number_input("Max Tokens", value=8192, step=100)
            
        with col2:
            st.subheader("翻译器选项")
            novels = st.text_input("EPUB 目录", value="./novels", help="The location of the epubs to translate")
            glossary = st.text_input("术语表", value="glossary.txt", help="Relative to the novels folder")
            chunk_size = st.number_input("分段数量", value=12, step=1)
            history = st.number_input("前文数量", value=12, step=1)
            attempts = st.number_input("重试次数", value=2, step=1)
            qa_pass = st.number_input("QA 次数", value=0, step=1)
        
        st.subheader("功能选项")
        col3, col4, col5 = st.columns(3)
        with col3:
            verbose = st.checkbox("详细日志", value=False)
            to_traditional = st.checkbox("繁体中文", value=True)
            chapter_abbrev = st.checkbox("扫描简称及昵称", value=False)
        with col4:
            glossary_only = st.checkbox("只生成术语表 (略过翻译)", value=False)
            add_glossary = st.checkbox("补充现有术语表", value=False)
        with col5:
            final_glossary = st.checkbox("只使用现有术语表 (跳过术语表扫描)", value=False)
            qa_only = st.checkbox("只进行 QA (略过翻译)", value=False)
            
        submit = st.form_submit_button("开始任务")
        
    if submit:
        now = datetime.now()
        time_string = now.strftime("%Y-%m-%d %H:%M:%S")
        st.info(f"{time_string} 任务开始，详细日志请查看终端控制台。")
        dict_file = os.path.join(novels, glossary)
        if not os.path.exists(novels):
            st.error(f"Please put your .epub files in the '{novels}' folder and try again.")
            return
            
        translator = TwoPassNovelTranslator(
            base_url=base_url,
            api_key=api_key,
            model_name=model, 
            dict_path=dict_file, 
            max_tokens=max_tokens, 
            verbose=verbose, 
            attempts=attempts,
            history=history,
            presence_penalty=presence_penalty,
            to_traditional=to_traditional,
            temperature=temperature,
            chapter_abbrev=chapter_abbrev
        )
        
        with st.spinner("Translating... check the terminal for real-time progress."):
            translator.start(
                novels, 
                qa_passes=qa_pass, 
                chunk_size=chunk_size, 
                qa_only=qa_only, 
                glossary_only=glossary_only, 
                add_glossary=add_glossary, 
                final_glossary=final_glossary
            )
        now = datetime.now()
        time_string = now.strftime("%Y-%m-%d %H:%M:%S")
        st.success(f"{time_string} 成功完成任务！")

app_cli = typer.Typer(help="EPUB Two-Pass Novel Translator")

@app_cli.command()
def main(
    novels: str = typer.Option("./novels", help="The location of the epubs to translate"),
    base_url: str = typer.Option("http://localhost:8080/v1", help="Base URL of the OpenAI-compatible server"),
    api_key: str = typer.Option("not-needed", help="API key for the server"),
    model: str = typer.Option("default", help="Model name to pass to the API"),
    webui: int = typer.Option(0, help="Start a Web UI on the specified port (e.g., 8000). Set to 0 to run in CLI mode."),
    verbose: bool = typer.Option(False, help="Enable verbose output during generation"),
    chunk_size: int = typer.Option(12, help="The chunk size for translation"),
    history: int = typer.Option(12, help="Number of previous translated sentences to send as history context"),
    max_tokens: int = typer.Option(8192, help="Maximum tokens to generate"),
    attempts: int = typer.Option(2, help="The number of attempts in translation_chunk when a pass fails"),
    temperature: float = typer.Option(1.0, help="Temperature for the model"),
    presence_penalty: float = typer.Option(0.0, help="Presence penalty to prevent looping (Note: high values may break JSON)"),
    glossary: str = typer.Option("glossary.txt", help="The name of the predefined dictionary file (relative to the novels folder)"),
    glossary_only: bool = typer.Option(False, help="Skip translation and QA, strictly run glossary extraction to generate final_glossary.json"),
    add_glossary: bool = typer.Option(False, help="Initialize global glossary from existing final_glossary.json before lore scanning"),
    final_glossary: bool = typer.Option(False, help="Initialize global glossary from existing final_glossary.json and skip lore scanning"),
    chapter_abbrev: bool = typer.Option(False, help="Run chapter abbreviation scanning"),
    qa_only: bool = typer.Option(False, help="Skip translation and strictly run QA passes on existing _zh.epub files"),
    qa_pass: int = typer.Option(0, help="Number of QA passes to run (Set to 0 to bypass)"),
    to_traditional: bool = typer.Option(True, help="Convert final translation output to Traditional Chinese before saving")
):
    if webui > 0:
        print(f"[*] Starting Web UI on port {webui}...")
        os.environ["STREAMLIT_MODE"] = "1"
        subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__), "--server.port", str(webui)])
        return

    dict_file = os.path.join(novels, glossary)
    
    if not os.path.exists(novels):
        os.makedirs(novels)
        print(f"Please put your .epub files in the '{novels}' folder and restart.")
        return

    translator = TwoPassNovelTranslator(
        base_url=base_url,
        api_key=api_key,
        model_name=model, 
        dict_path=dict_file, 
        max_tokens=max_tokens, 
        verbose=verbose, 
        attempts=attempts,
        history=history,
        presence_penalty=presence_penalty,
        to_traditional=to_traditional,
        temperature=temperature,
        chapter_abbrev=chapter_abbrev
    )
    translator.start(novels, qa_passes=qa_pass, chunk_size=chunk_size, qa_only=qa_only, glossary_only=glossary_only, add_glossary=add_glossary, final_glossary=final_glossary)

if __name__ == "__main__":
    if os.environ.get("STREAMLIT_MODE") == "1":
        run_streamlit_ui()
    else:
        app_cli()