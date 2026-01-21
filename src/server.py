import time
import uuid
import random
from typing import List

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from transformers import AutoTokenizer
from dataclasses import dataclass

# =========================================================
# FastAPI App
# =========================================================
app = FastAPI()

# # =========================================================
# # Tokenizer
# # =========================================================
# tokenizer = AutoTokenizer.from_pretrained(
#     r"C:\Users\uh\study\opt-350m",
#     use_fast=True
# )
#
# VOCAB_SIZE = tokenizer.vocab_size

# =========================================================
# OpenAI Request Models (Pydantic)
# =========================================================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 128
    top_p: float = 0.9
    temperature: float = 0.7
    stream: bool = False


# =========================================================
# OpenAI Response Models (dataclass)
# =========================================================
@dataclass
class ChatCompletionMessage:
    role: str
    content: str


@dataclass
class ChatCompletionChoice:
    index: int
    message: ChatCompletionMessage
    finish_reason: str


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletionResponse:
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

# =========================================================
# Chat Completions API
# =========================================================
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # # 拼接 prompt（模拟）
    # prompt = ""
    # for msg in req.messages:
    #     prompt += f"{msg.role}: {msg.content}\n"

    # # 解码
    # output_text = tokenizer.decode(
    #     output_tokens,
    #     skip_special_tokens=True
    # )
    #
    # prompt_tokens = len(tokenizer.encode(prompt))
    # completion_tokens = len(output_tokens)

    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens
    )

    choice = ChatCompletionChoice(
        index=0,
        message=ChatCompletionMessage(
            role="assistant",
            content=output_text
        ),
        finish_reason="length"
    )

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=req.model,
        choices=[choice],
        usage=usage
    )

    return jsonable_encoder(response)


# =========================================================
# Health Check
# =========================================================
@app.get("/health")
async def health():
    return {"status": "ok"}


# # =========================================================
# # Run Server (python server.py)
# # =========================================================
# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.step_loop(
#         app,
#         host="0.0.0.0",
#         port=8000,
#         reload=False
#     )
