"""对话模型工厂：OpenAI 兼容接口，指向配置的 base_url（智谱/DeepSeek/…可切换）。"""

from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings


def build_chat_model(settings: Settings | None = None, *, streaming: bool = False, temperature: float | None = None) -> ChatOpenAI:
    """streaming=True 供 generate 节点流出 token；判分/改写用非流式。"""
    s = settings or get_settings()
    return ChatOpenAI(
        model=s.chat_model,
        base_url=s.chat_api_base_url,
        api_key=s.chat_api_key,
        temperature=s.chat_temperature if temperature is None else temperature,
        streaming=streaming,
        timeout=60,
        max_retries=2,
    )
