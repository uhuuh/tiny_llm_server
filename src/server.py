import time
import uuid
import random
from typing import List

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from transformers import AutoTokenizer
from dataclasses import dataclass

app = FastAPI()

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

@dataclass
class ChatCompletionRequestResult:
    id: str
    prompt_tokens: List[int]
    completion_tokens: List[int]
    output_text: str

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    res: ChatCompletionRequestResult = await app.state.chat_completions_handler(req)

    usage = Usage(
        prompt_tokens=res.prompt_tokens,
        completion_tokens=res.completion_tokens,
        total_tokens=res.prompt_tokens + res.completion_tokens
    )

    choice = ChatCompletionChoice(
        index=0,
        message=ChatCompletionMessage(
            role="assistant",
            content=res.output_text
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

@app.get("/health")
async def health():
    return {"status": "ok"}


