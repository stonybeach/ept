# **EPUB 三阶段日文小说翻译器用户指南**

这是一个专为日文轻小说设计的 EPUB 翻译工具。它支持通过 OpenAI 兼容的 API 后端（如 mlx\_lm.server、vLLM、llama.cpp、youssofal/mtplx 等）来进行上下文连贯的批量翻译，并且提供 Streamlit 网页图形界面（Web UI）。

## **1\. 工作流程 (Workflow)**

此脚本采用三阶段架构，包括可选的质量保证（QA）校对阶段。三个阶段可以一次进行，或者按需要分开执行。分开执行时可以按需要使用不同 LLM 模型。

\[原始 EPUB\] \+ \[绿站 glossary.txt\]  
       ↓  
阶段 1：术语扫描与实体提取 (Lore Scanning)     → 生成 \[final\_glossary.json\]  
       ↓  
阶段 2：上下文感知批量翻译 (Translation)       → 生成 \[\*\_zh.epub\]  
       ↓  
阶段 3：自动化质量保证校对 (Quality Assurance) → 生成 \[\*\_final.epub\]  

### **阶段 1：术语扫描与实体提取 (Lore Scanning)**

* **目的**：在翻译开始前，扫描全书提取关键的人名、地名、特别物品和活动等专有名词。  
* **输入文件**：  
  * **原始 EPUB 文件**：放置于 \--novels（默认 ./novels）目录下的日文 .epub 小说。  
  * **预定义词典文件**（可选）：用户手动编辑或绿站提供的术语映射表，通常命名为 glossary.txt（格式为 日文原名 \=\> 中文译名）。  
* **输出文件**：  
  * **final\_glossary.json**：扫描完成后自动生成的全书主术语表。如果提供了预定义词典，它会与自动提取的术语融合成该主术语表。

### **阶段 2：上下文感知批量翻译 (Translation)**

* **目的**：利用上阶段生成的术语表，以分段批量（Batch/Chunk）的方式翻译小说主体，同时维护滑动上下文窗口（历史翻译记录）以保证人称、语气和叙事逻辑的前后一致。  
* **输入文件**：  
  * **原始 EPUB 文件**。  
  * **final\_glossary.json**（上阶段生成的术语表）。  
* **输出文件**：  
  * **\*\_zh.epub**：翻译完成后的临时 EPUB 文件。此阶段会将原文段落保留，并紧随其后插入对应翻译，以便阅读校对或供下一阶段 QA 评估。

### **阶段 3：自动化质量保证校对 (QA Pass)**

* **目的**：自动校对翻译结果，修正由于大模型在批量翻译时可能出现的漏翻、错翻、术语未严格对齐等细节问题。  
* **输入文件**：  
  * **\*\_zh.epub**：阶段 2 生成的带双语对比的 EPUB 文件。  
  * **final\_glossary.json**：主术语表。  
* **输出文件**：  
  * **\*\_final.epub**：经过精细校对后的最终 EPUB 文件。所有被 QA 修改过的翻译都会在此阶段被完美更新进电子书中。

## **2\. 选项介绍 (Options)**

无论您是在命令行（CLI）中运行此脚本，还是在图形界面（Web UI）中使用，以下选项都将控制翻译器的行为：

### **网络与 API 配置**

* **\--base-url** *(默认: http://localhost:8080/v1)*  
  * 指向 OpenAI 兼容 API 后端的链接。适用于各种本地运行的推理框架（如 mlx\_lm.server、llama.cpp）或云端 API（如 OpenRouter、DeepSeek 官网 API）。  
* **\--api-key** *(默认: not-needed)*  
  * API 认证密钥。本地推理服务器通常不需要，若使用第三方云服务则填入相应的 Key（在 Web UI 中会以密码模式脱敏显示）。  
* **\--model** *(默认: default)*  
  * 需要调用的模型名称字符串（例如 deepseek-chat、models/gemma-4-31b-it-8bit，视 API 而定）。  
* **\--temperature** *(默认: 1.0)*  
  * 采样温度。较低的值（如 0.0）会使模型输出更加确定和严谨；较高的值（如 1.5）会带来更具创造力和丰富句式的文学化翻译。  
* **\--presence-penalty** *(默认: 0.0)*  
  * 存在惩罚系数。设置为大于 0（如 1.0 至 1.5）的值可以有效阻止模型在翻译过程中出现无限循环。如果使用 Qwen 3.6，建议设为 1.5。

### **翻译性能与资源调节**

* **\--chunk-size** *(默认: 12\)*  
  * 每次打包发送给模型进行翻译的日文段落数量。设置太小会增加 API 请求频率和前文重读开销；设置过大可能会超出单次生成的最大 Token 限制或导致格式崩坏。建议保持在 12 左右。  
* **\--history** *(默认: 12\)*  
  * 翻译当前批次（Chunk）时，作为上下文发送给模型的历史中文译文行数。利用滑动窗口技术，让模型“看着前文翻后文”，保证时态和人称不突兀。  
* **\--max-tokens** *(默认: 8192\)*  
  * 限制模型单次 API 响应生成的最大 Token 数量。  
* **\--attempts** *(默认: 2\)*  
  * 遇到翻译失败（如翻译返回的段落数量不符、或者返回的内容依然包含大量日文未被正常翻译）时的重试次数上限。如果多次重试仍失败，脚本会自动退化为极其稳妥的“单行挨个翻译”模式。

### **功能与开关配置**

* **\--webui** *(默认: 0\)*  
  * **启用图形界面**。指定一个端口（例如 \--webui 8000），即可启动 Web 浏览器操作界面。设为 0 则代表纯 CLI 命令行模式。  
* **\--verbose** *(默认: 关闭)*  
  * 开启详细日志。启用后会在控制台里实时打印模型返回的完整原始响应，方便调试。  
* **\--to-traditional** *(默认: 启用)*  
  * 将翻译终稿一键转化为繁体中文。如果关闭此项，将保存为简体中文。  
* **\--chapter-abbrev** *(默认: 关闭)*  
  * 针对每章最开始的 2000 字运行一次局部“简称与昵称”提取，防止模型将书中的人物外号或局部特定昵称翻译错。  
* **\--glossary** *(默认: glossary.txt)*  
  * 放在小说目录下的预定义术语文本文件名。

### **阶段执行控制开关**

* **\--glossary-only** *(默认: 关闭)*  
  * 脚本运行到阶段 1（术语生成并保存到 final\_glossary.json）后立即退出，不启动任何翻译任务，方便有需要时进行人工检查。  
* **\--add-glossary** *(默认: 关闭)*  
  * 在阶段 1 启动时，先加载现有的 final\_glossary.json 文件作为基底，再往里扫描补充新发现的词汇，方便翻译新卷时继承前卷的术语表。  
* **\--final-glossary** *(默认: 关闭)*  
  * **跳过阶段 1 的扫描**。直接读取现存的 final\_glossary.json 开始翻译。这在您已经手工打磨好完美术语表时节省时间。  
* **\--qa-only** *(默认: 关闭)*  
  * 直接跳过阶段 1 和 阶段 2。仅对当前小说目录下的 \*\_zh.epub 文件执行阶段 3 的 QA 校对。  
* **\--qa-pass** *(默认: 0\)*  
  * 指定 QA 校对的轮数。设置为 0 表示不运行 QA 阶段；设为 1 或更多则会在翻译结束后自动运行指定轮数的自动校正。

## **3\. 使用实例 (Command Line Examples)**

以下是两组典型的命令行调用范例以及它们背后参数的设计原理：

### **示例 1**

python translate\_epubs\_new.py \--temperature 1.5 \--model models/gemma-4-31b-it-8bit

* **运行分析**：  
  * **\--model models/gemma-4-31b-it-8bit**：指定本地运行的 Gemma 4 31B 的 8-bit 量化版本。Gemma 4 模型需要另外以 llama.cpp、mlx_lm 等工具以 models/gemma-4-31b-it-8bit 为名字在本地执行。
  * **\--temperature 1.5**：由于 Gemma 4 有时会使用英语输出部分词语，此配置可以减低出错机会。  

MLX 后端使用示例：

mlx_lm.server --model models/gemma-4-31b-it-4bit --prefill-step-size 4096 --chat-template-args='{"enable_thinking": false}' 

### **示例 2**

python translate\_epubs\_new.py \--presence-penalty 1.5 \--model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed

* **运行分析**：  
  * **\--model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed**：指定本地运行的 Qwen 3.6 27B 的 4-bit 量化版本。Qwen 3.6 模型在本地使用 MTPLX 以 Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed 为名字执行。
  * **\--max-tokens 16384**：由于启用推理，token 量要高一点。
  * **\--presence-penalty 1.5**：使用 Qwen 3.6 如果不设置 1.5 的存在惩罚（Presence Penalty），有可能会出现模型持续输出不重复的新内容，进入无限复读循环。

MTPLX 后端使用示例：

mtplx serve \
  --host localhost \
  --port 8080 \
  --model Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed \
  --max-tokens 16384 \
  --profile sustained \
  --mtp \
  --depth 3 \
  --reasoning on \
  --reasoning-parser qwen3 \
  --no-stats-footer

### **示例 3**

python translate_epubs_new.py --max-tokens 16384 --model deepseek-reasoning --history 5 --temperature 0.5 --chunk-size 50

* **运行分析**：  
  * **\--model deepseek-reasoning**：指定本地运行的 deepseek 推理模型。
  * **\--max-tokens 16384**：由于启用推理，token 量要高一点。
  * **\--temperature 0.5**：推荐使用 0.5，如果太高会导致不遵守术语表，太低则输出格式不正确。

Deepseek v4 后端使用示例：

./ds4-server --ctx 32768 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 16384 --port 8080

## **4\. 启动图形操作界面**

如果您不喜欢在终端中输入繁琐的参数，可以使用以下命令启动内置的 Web 控制面板：

python translate\_epubs\_new.py \--webui 8000

启动后，在浏览器中打开 http://localhost:8000 即可在可视化的网页窗口中调整所有参数，并启动您的轻小说翻译任务。