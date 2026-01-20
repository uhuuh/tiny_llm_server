from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import sys

# =========================================================
# Client Config
# =========================================================
BASE_URL = "http://localhost:8000/v1"
API_KEY = "test"
MODEL_NAME = "fake-llm"
DEFAULT_MAX_TOKENS = 100

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

# =========================================================
# Core Chat Request (唯一出口)
# =========================================================
def chat_request(
    messages: List[Dict[str, str]],
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> str:
    """
    所有 chat completion 都从这里走
    """
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content


# =========================================================
# 单条请求
# =========================================================
def single_request():
    messages = [
        {"role": "user", "content": "你好，介绍一下你自己"}
    ]
    print(chat_request(messages, max_tokens=50))


# =========================================================
# 多 batch（并发）
# =========================================================
def batch_requests():
    prompts = [
        "你好，介绍一下你自己",
        "什么是 FastAPI？",
        "解释一下 Transformer",
        "什么是大模型推理？"
    ]

    def run(prompt: str) -> str:
        return chat_request(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, prompts))

    for i, r in enumerate(results):
        print(f"[Batch {i}] {r}")


# =========================================================
# 交互式聊天（多轮对话）
# =========================================================
def interactive_chat():
    messages: List[Dict[str, str]] = []

    print("进入交互模式，输入 exit / quit 退出\n")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})

        assistant_msg = chat_request(messages)
        print("Assistant:", assistant_msg)

        messages.append({
            "role": "assistant",
            "content": assistant_msg
        })


# =========================================================
# Main
# =========================================================
def main():
    modes = {
        "1": single_request,
        "2": batch_requests,
        "3": interactive_chat
    }

    print(
        "\n选择运行模式：\n"
        "1 - 单条请求\n"
        "2 - 多 batch 并发请求\n"
        "3 - 交互式聊天\n"
    )

    choice = input("请输入 1 / 2 / 3: ").strip()

    fn = modes.get(choice)
    if fn is None:
        print("无效输入")
        sys.exit(1)

    fn()


if __name__ == "__main__":
    main()
