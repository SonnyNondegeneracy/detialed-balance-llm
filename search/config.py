import re

api_path = "./api_keys.json"
output_path = "./search/database"
total_cycles = 48           # 总运行轮数
island_num = 12             # 岛屿数
examples_num= 5
sample_temperature=15.0
model_sample_temperature=15.0
model_assess_window_size=5
hand_over_threshold=0.1
model_assess_average_order=15.0

N_online = 3                # 在线运行轮数
N_offline = 0               # 离线运行轮数
online_cycle_epochs = 10    # 在线轮数
offline_cycle_epochs = 40   # 离线轮数
online_models = [
    'gpt-5-mini', 'gpt-5', 'qwen3', 'qwen-plus', 'gemini-2.5-flash', 'grok-4', 'doubao', 'gemini-2.5-pro', "claude-4"
    ]                       # 在线模型
offline_models = [
    'gpt4-nano', 'gpt5-nano', 'gemini-2.0-flash', 'doubao-lite', 'glm-4-flash', 'grok-3-mini', 'qwen-turbo', 'gemini-2.5-09-2025', 'glm-4-flash1'
    ]                       # 离线模型
online_translator = 'gpt5-nano'            # 在线翻译器
offline_translator = 'gemini-2.5-09-2025'   # 离线翻译器

initial_ideas = [
    "一个可能合适的势应该将函数f的势与其复杂度和包含的数学操作相关联。例如，可以设计一个势函数，使得包含更多复杂操作（如三角函数、指数函数等）的函数具有更低的势值，而简单的多项式函数则具有较高的势值。这种设计暗示模型倾向于生成更复杂的函数，从而探索更广泛的函数空间。",
    "<potential>def potential(f):\n    return 0.0</potential>",
]

system = """
你是一个高级AI助手，专门负责设计势泛函以优化基于大模型的agent在nikuradse_2拟合任务中的搜索效率。你的任务是生成创新且有效的势泛函，这些泛函将指导agent在复杂的数学函数空间中进行探索和拟合。请确保你的势泛函能够捕捉到agent生成中的特性，并促进其发现高质量的拟合函数。
"""

epilogue = r"""
势potential(f)是一个泛函，它接受一个表示数学函数f的字符串（例如"param1 - param2 * log_v_k_nu - param3 / (1 + exp(-param4 * (log_v_k_nu - param5)))"）作为输入，并输出一个实数值，表示该函数对大模型构造的agent在给定的拟合任务：nikuradse_2中的搜索潜力。这个实数越低，表明该函数越靠近大模型生成的下游，为了说明问题，我们把$\mathcal{T}(f,g)$定义为agent接受f时，输出g的概率。那么势函数的目标是最小化以下期望值：
\begin{align}
  \mathcal{S} = \sum_{f\text{ as input}}\int_{g\text{ as output}} \mathcal{T}(f,g)\, \left(1 - V_{\mathcal{T}}(f) + V_{\mathcal{T}}(g)\right)^2
\end{align}
为了进一步说明问题，以下是一些可能的势函数示例：
"""

prologue = r"""
在这个任务中，你的目标是设计一个势函数potential(f)，它能够有效地引导一个基于大模型的agent在nikuradse_2拟合任务中进行搜索。势函数接受一个表示数学函数f的字符串作为输入，并输出一个实数值，表示该函数在agent搜索过程中的产生新函数的潜力。势函数的目标是最小化agent生成的下游函数的势能，从而提高搜索效率和结果质量。
现在请你设计一个合适的势函数potential(f)，并将结果包裹在<potential>...</potential>标签中返回。注意字符串可能有非交换性质，比如 param1 * tanh(param2 * x) + param3 和 param1 + param2 * tanh(param3 * x) 字符串的势函数可能不同。
"""

def filter(string: str) -> str:
    """
    过滤器函数，用于从论述中提取势函数的表达式。
    """
    pattern = r'<potential>(.*?)</potential>'
    match = re.search(pattern, string, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return "【未找到势函数表达式，请参考论述】"