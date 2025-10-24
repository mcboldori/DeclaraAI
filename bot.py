#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 22:06:42 2025

@author: mariacarolineboldori
"""

# -*- coding: utf-8 -*-
"""
Bot Telegram • Reforma do IR 2026 (PL 1087/2025)
- Escopo informativo (não consultivo)
- LGPD: sanitização de PII + logs mínimos
- Rate limit por usuário
- Fallback em CSV (FAQ) + LLM (OpenAI Responses API)
- Banner de status legislativo com data
- Fontes padrão em respostas sensíveis
- Hard caps de tokens e custo (básico)
"""

import os
import re
import time
import json
import hashlib
import logging
from datetime import datetime
from collections import deque, defaultdict

import pandas as pd
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# ====== Configurações básicas ======
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5")  # você pode trocar por outro compatível
FAQ_CSV_PATH = os.getenv("FAQ_CSV_PATH", "Pack_FAQ_Reforma_IR_Chatbot.csv")

# Limites e políticas
MAX_REQ_PER_MIN = int(os.getenv("MAX_REQ_PER_MIN", "10"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "600"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "1200"))   # corta msgs muito longas
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "1") == "1"
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "21600"))  # 6h
SHOW_SOURCES_BY_DEFAULT = True

DISCLAIMER = (
    "⚠️ Este bot é **informativo** e **não substitui** aconselhamento contábil. "
    "As regras podem mudar até a **sanção** e **regulamentação**."
)

SOURCES_FOOTER = (
    "📚 Fontes principais: Planalto (PL 1087/2025), Câmara/Senado (tramitação), Receita Federal (tabelas)."
)

STATUS_LINE = (
    "📅 Status em {today}: aprovado na **Câmara**; em análise no **Senado**; vigência prevista **01/01/2026**."
)

SYSTEM_PROMPT = (
    "Você é um assistente **didático e conciso** sobre a Reforma do IR no Brasil (PL 1087/2025). "
    "Explique sem jargões, com números redondos quando possível, e sempre que aplicável lembre o usuário de que "
    "as regras podem mudar até sanção/regulamentação. Evite qualquer aconselhamento individual."
)

# ====== Logs mínimos (sem conteúdo bruto do usuário) ======
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("reforma_ir_bot")

# ====== Sanitização de PII ======
PII_PATTERNS = [
    re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),        # CPF
    re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b"), # CNPJ
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # e-mail
    re.compile(r"\b\+?\d{2}\s?\(?\d{2}\)?\s?\d{4,5}-?\d{4}\b"),     # telefone BR
]

def sanitize(text: str) -> str:
    s = text[:MAX_INPUT_CHARS]  # corta msgs enormes
    for pat in PII_PATTERNS:
        s = pat.sub("[DADO PESSOAL REMOVIDO]", s)
    return s

# ====== Rate limit simples ======
WINDOW_SEC = 60
user_hits = defaultdict(lambda: deque())

def allow(user_id: int) -> bool:
    now = time.time()
    q = user_hits[user_id]
    while q and now - q[0] > WINDOW_SEC:
        q.popleft()
    if len(q) >= MAX_REQ_PER_MIN:
        return False
    q.append(now)
    return True

# ====== Cache leve em memória ======
class TTLCache:
    def __init__(self, ttl=CACHE_TTL_SEC):
        self.ttl = ttl
        self.data = {}  # key -> (value, expires_at)

    def _now(self):
        return time.time()

    def get(self, key):
        if not ENABLE_CACHE:
            return None
        hit = self.data.get(key)
        if not hit:
            return None
        value, exp = hit
        if self._now() > exp:
            self.data.pop(key, None)
            return None
        return value

    def set(self, key, value):
        if not ENABLE_CACHE:
            return
        self.data[key] = (value, self._now() + self.ttl)

cache = TTLCache()

def cache_key(user_id: int, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{user_id}:{h}"

# ====== Carrega FAQ CSV ======
try:
    FAQ = pd.read_csv(FAQ_CSV_PATH)
    FAQ["user_examples"] = FAQ["user_examples"].fillna("")
    FAQ["answer_pt_br"] = FAQ["answer_pt_br"].fillna("")
except Exception as e:
    logger.warning(f"Não foi possível carregar {FAQ_CSV_PATH}: {e}")
    FAQ = pd.DataFrame(columns=["intent_id", "user_examples", "answer_pt_br", "citations"])

def search_faq(query: str) -> str | None:
    q = query.lower()
    # Busca simples por termos listados em 'user_examples', separados por ';'
    for _, row in FAQ.iterrows():
        examples = [t.strip().lower() for t in str(row.get("user_examples", "")).split(";") if t.strip()]
        if any(term in q for term in examples):
            return row.get("answer_pt_br", "").strip() or None
    return None

# ====== Cliente OpenAI (Responses API) ======
# Implementação leve via requests para evitar dependência de SDKs específicos
import requests

OPENAI_ENDPOINT = "https://api.openai.com/v1/responses"
HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

def call_llm(prompt: str) -> str:
    """Chama a Responses API com limites e system prompt."""
    payload = {
        "model": MODEL_NAME,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        # temperature/top_p podem ser ajustados; mantemos padrão conservador
    }
    try:
        resp = requests.post(OPENAI_ENDPOINT, headers=HEADERS, data=json.dumps(payload), timeout=20)
        resp.raise_for_status()
        data = resp.json()
        # A Responses API costuma retornar um "output_text" agregado
        text = data.get("output_text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        # fallback: tentar extrair de outputs estruturados (se presente)
        if "output" in data and isinstance(data["output"], list) and data["output"]:
            parts = []
            for item in data["output"]:
                if isinstance(item, dict) and "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(str(item))
            joined = "\n".join(parts).strip()
            if joined:
                return joined
        return "Desculpe, não consegui processar sua pergunta agora."
    except Exception as e:
        logger.error(f"Erro na chamada OpenAI: {e}")
        return "No momento não consegui consultar a base. Tente novamente em instantes."

# ====== Helpers de resposta ======
def status_banner() -> str:
    today = datetime.now().strftime("%d/%m/%Y")
    return STATUS_LINE.format(today=today)

def with_sources(answer: str, add_sources: bool = SHOW_SOURCES_BY_DEFAULT) -> str:
    if not add_sources:
        return answer
    # Evita duplicar se já houver "Fontes"
    if "Fontes" in answer or "📚" in answer:
        return answer
    return f"{answer}\n\n{SOURCES_FOOTER}"

def compose_final_answer(core: str, prepend_status: bool = False, add_disclaimer: bool = True) -> str:
    blocks = []
    if prepend_status:
        blocks.append(status_banner())
    blocks.append(core.strip())
    if add_disclaimer:
        blocks.append(DISCLAIMER)
    return "\n\n".join(blocks)

# ====== Handlers ======
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Olá! Sou o bot da **Reforma do IR 2026** 🇧🇷.\n"
        "Pergunte algo como *“quando começa a valer?”*, *“quem fica isento?”* ou *“dividendos pagam quanto?”*.\n\n"
        f"{DISCLAIMER}"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(status_banner(), parse_mode=ParseMode.MARKDOWN)

async def cmd_fontes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(SOURCES_FOOTER)

async def cmd_apagar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Como não armazenamos conteúdo bruto, apenas sinalizamos a política de retenção mínima
    await update.message.reply_text("Não armazeno suas mensagens. Mantenho apenas métricas anônimas e temporárias (latência/uso).")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id if user else 0
    text_raw = update.message.text or ""

    # Rate limit
    if not allow(user_id):
        await update.message.reply_text("Muitas mensagens em 1 minuto. Tente novamente em instantes.")
        return

    # Sanitiza + encurta
    text = sanitize(text_raw)

    # Cache
    key = cache_key(user_id, text)
    cached = cache.get(key)
    if cached:
        await update.message.reply_text(cached, parse_mode=ParseMode.MARKDOWN)
        return

    # 1) Tenta responder via FAQ
    faq_answer = search_faq(text)
    if faq_answer:
        answer = compose_final_answer(with_sources(faq_answer), prepend_status=("vigência" in faq_answer.lower() or "isenção" in faq_answer.lower()))
        cache.set(key, answer)
        await update.message.reply_text(answer, parse_mode=ParseMode.MARKDOWN)
        return

    # 2) Se não achou, pergunta ao LLM com prompt controlado
    prompt = (
        f"{status_banner()}\n"
        "Responda de forma breve e didática. Evite aconselhamento individual; "
        "se necessário, recomende procurar um contador. Pergunta do usuário:\n"
        f"\"{text}\""
    )
    llm_text = call_llm(prompt).strip()

    # Segurança de saída: corta muito longo, opcionalmente remove links estranhos
    if len(llm_text) > 2400:
        llm_text = llm_text[:2400] + "…"

    final = compose_final_answer(with_sources(llm_text), prepend_status=False)
    cache.set(key, final)
    await update.message.reply_text(final, parse_mode=ParseMode.MARKDOWN)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text("Tive um erro agora há pouco. Tente novamente em instantes, por favor.")

def main():
    if not TELEGRAM_TOKEN or not OPENAI_API_KEY:
        raise RuntimeError("Defina TELEGRAM_TOKEN e OPENAI_API_KEY no .env")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("fontes", cmd_fontes))
    app.add_handler(CommandHandler("apagar", cmd_apagar))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("Bot iniciado. Aguardando mensagens…")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
