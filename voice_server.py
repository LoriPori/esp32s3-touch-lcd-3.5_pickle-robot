import os
import re
import io
import json
import uuid
import tempfile
import subprocess
from difflib import SequenceMatcher
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request, Response, Header, HTTPException, Depends
from pydantic import BaseModel

from groq import Groq
from google import genai
from google.genai import types

import edge_tts

app = FastAPI()

# ==========================================
# Configuração de Chaves API e Clientes
# ==========================================
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
PICKLE_SHARED_SECRET = os.environ.get("PICKLE_SHARED_SECRET", "")

def verify_secret(x_pickle_secret: str = Header(default="")):
    if not PICKLE_SHARED_SECRET or x_pickle_secret != PICKLE_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Não autorizado")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

GEMINI_MODEL = "gemini-3.1-flash-lite"
GROQ_STT_MODEL = "whisper-large-v3-turbo"

ACCEPTED_VARIANTS = [
    "pickle", "picle", "pico", "pika", "pica", 
    "pekle", "pikl", "becle", "piclo", "pizzel"
]

# Alucinações frequentes do Whisper em momentos de ruído/silêncio
WHISPER_SILENCE_HALLUCINATIONS = {
    "obrigado", "obrigada", "obrigado.", "obrigada.",
    "subscreva", "inscreva-se", "deixe o seu like",
    "amém", "amém.", "obrigado por assistir", "legendas:",
    "obrigado pela vossa atenção", "já está", "tchau"
}

# [ALTERAÇÃO] Dicionário de correções fonéticas específicas para o Whisper em PT-PT
COMMON_PHRASE_CORRECTIONS = {
    r'\bponto\s+chumas\b': 'como te chamas',
    r'\bvamos\s+juntos\s*,\s*chamas\b': 'como te chamas',
    r'\bchumas\b': 'chamas',
    r'\bcom\s+te\s+chamas\b': 'como te chamas',
    r'\bquem\s+es\s+tu\b': 'quem és tu',
    r'\bque\s+e\s+isso\b': 'o que é isso'
}

def fix_stt_phonetics(text: str) -> str:
    corrected = text
    for pattern, replacement in COMMON_PHRASE_CORRECTIONS.items():
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
    return corrected

def contains_speech_vad(audio_bytes: bytes) -> bool:
    try:
        import webrtcvad
        vad = webrtcvad.Vad(3)
        sample_rate = 16000
        frame_duration = 30
        frame_size = int(sample_rate * (frame_duration / 1000.0) * 2)

        speech_frames = 0
        total_frames = 0

        for i in range(0, len(audio_bytes) - frame_size, frame_size):
            frame = audio_bytes[i:i + frame_size]
            if len(frame) == frame_size:
                total_frames += 1
                if vad.is_speech(frame, sample_rate):
                    speech_frames += 1

        if total_frames == 0:
            return True
        return (speech_frames / total_frames) > 0.15
    except Exception:
        return True

# ==========================================
# Gestão de Ficheiros e Memória
# ==========================================
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickle_memory.json")
REMINDERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickle_reminders.json")

def load_reminders():
    if not os.path.exists(REMINDERS_FILE):
        return []
    try:
        with open(REMINDERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Lembretes] Erro ao carregar: {e}")
        return []

def save_reminders(reminders_list):
    with open(REMINDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(reminders_list, f, ensure_ascii=False, indent=2)

def load_memory() -> list:
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Memória] Erro ao carregar: {e}")
        return []

def save_memory(facts: list):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)

reminders = load_reminders()
memory_facts = load_memory()

print(f"[Lembretes] {len(reminders)} lembrete(s) carregado(s)")
print(f"[Memória] {len(memory_facts)} facto(s) carregado(s)")

# ==========================================
# Funções Auxiliares de Tratamento de Texto
# ==========================================
def clean_portuguese_text(text: str) -> str:
    if re.search(r'[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\u4e00-\u9faf]', text):
        return ""
    cleaned = re.sub(r'[^a-zA-Z0-9áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ\s\.,!\?\'-]', '', text)
    return cleaned.strip()

def is_hallucination(text: str) -> bool:
    """Deteta repetições consecutivas de palavras e frases curtas geradas por ruído."""
    words = text.lower().split()
    if not words:
        return False

    current_repeat = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            current_repeat += 1
            if current_repeat >= 3:
                return True
        else:
            current_repeat = 1

    if len(words) >= 3 and len(set(words)) == 1:
        return True

    clean_text = " ".join(words)
    if re.search(r'(\b\w+\s+\w+\b)(?:\s+\1){2,}', clean_text):
        return True

    return False

def map_to_pickle(text: str) -> str:
    words = text.split()
    for i, word in enumerate(words):
        clean_word = re.sub(r'[^\w\s]', '', word.lower())
        matched = clean_word in ACCEPTED_VARIANTS
        if not matched:
            similarity = SequenceMatcher(None, clean_word, "pickle").ratio()
            matched = similarity >= 0.78
        if matched:
            remainder = words[i + 1:]
            return "Pickle " + " ".join(remainder)
    return text

# ==========================================
# Endpoints da API
# ==========================================

@app.post("/stt", dependencies=[Depends(verify_secret)])
async def stt(request: Request):
    audio_bytes = await request.body()
    
    if not contains_speech_vad(audio_bytes):
        print("[STT Ignorado]: Ruído de fundo sem presença de voz")
        return {"text": ""}

    try:
        # [ALTERAÇÃO] Prompt expandido com frases chave comuns para guiar o Whisper
        transcription = groq_client.audio.transcriptions.create(
            file=("audio.wav", io.BytesIO(audio_bytes), "audio/wav"),
            model=GROQ_STT_MODEL,
            prompt="Transcrição em português de Portugal para o robô Pickle. Perguntas comuns: Como te chamas?, Olá Pickle, Quem és tu?, Que horas são?, O que podes fazer?.",
            response_format="json",
            language="pt",
            temperature=0.0
        )
        raw_text = transcription.text.strip()
    except Exception as e:
        print(f"[Groq STT ERRO]: {e}")
        return {"text": ""}

    normalized_check = raw_text.lower().strip(' .!?,\n\t')
    if not normalized_check or normalized_check in WHISPER_SILENCE_HALLUCINATIONS:
        print(f"[STT Ignorado]: Ruído interpretado como silêncio/alucinação ('{raw_text}')")
        return {"text": ""}

    if is_hallucination(raw_text):
        print(f"[STT Ignorado]: Alucinação por repetição detetada ('{raw_text}')")
        return {"text": ""}
    
    portuguese_text = clean_portuguese_text(raw_text)
    if not portuguese_text:
        print(f"[STT Ignorado]: Ruído ou caracteres inválidos ('{raw_text}')")
        return {"text": ""}

    # [ALTERAÇÃO] Aplicação do filtro de correção fonética antes do envio
    fixed_text = fix_stt_phonetics(portuguese_text)

    normalized_text = map_to_pickle(fixed_text)
    print(f"[PT-Bruto]: '{raw_text}' -> [Corrigido]: '{fixed_text}' -> [Normalizado]: '{normalized_text}'")
    return {"text": normalized_text}


class TtsRequest(BaseModel):
    text: str

@app.post("/tts", dependencies=[Depends(verify_secret)])
async def tts(req: TtsRequest):
    clean_text = clean_portuguese_text(req.text)
    if not clean_text:
        return Response(content=b"", media_type="audio/wav")

    tmp_mp3 = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
    tmp_wav = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")

    try:
        communicate = edge_tts.Communicate(clean_text, "pt-PT-DuarteNeural", rate="+5%")
        await communicate.save(tmp_mp3)

        subprocess.run([
            "ffmpeg", "-y", "-i", tmp_mp3, 
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", tmp_wav
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        with open(tmp_wav, "rb") as f:
            wav_data = f.read()

        return Response(content=wav_data, media_type="audio/wav")

    except Exception as e:
        print(f"[Edge-TTS ERRO]: {e}")
        return Response(content=b"", media_type="audio/wav", status_code=500)
    finally:
        if os.path.exists(tmp_mp3):
            os.remove(tmp_mp3)
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


@app.post("/chat", dependencies=[Depends(verify_secret)])
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    last_user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_message = msg.get("content", "").strip()
            break

    print(f"[Chat] Última mensagem do utilizador: '{last_user_message}'")
    lower_msg = last_user_message.lower()

    remember_match = re.match(r'^lembra[\s\-,]*te[\s,]*(?:que\s+)?', lower_msg)
    if remember_match:
        new_fact = last_user_message[remember_match.end():].strip(" ,.")
        if new_fact:
            memory_facts.append(new_fact)
            save_memory(memory_facts)
            print(f"[Memória] Novo facto guardado: '{new_fact}'")
            return {"message": {"content": f"Ok, vou lembrar-me disso: {new_fact}."}}

    if re.match(r'^esquece\s+tudo', lower_msg):
        memory_facts.clear()
        save_memory(memory_facts)
        print("[Memória] Memória apagada por pedido do utilizador")
        return {"message": {"content": "Pronto, esqueci tudo o que sabia sobre ti."}}

    requested_model = body.get("model", GEMINI_MODEL)

    reminder_match = re.match(
        r'^lembra[\s\-,]*me\s+de\s+(.+?)\s+[àa]s?\s+(\d{1,2})[:h](\d{2})?',
        last_user_message, re.IGNORECASE
    )
    if reminder_match:
        task = reminder_match.group(1).strip(" ,.")
        hour = int(reminder_match.group(2))
        minute = int(reminder_match.group(3)) if reminder_match.group(3) else 0
        target_time = f"{hour:02d}:{minute:02d}"
        reminders.append({"task": task, "time": target_time, "delivered": False})
        save_reminders(reminders)
        print(f"[Lembretes] Novo lembrete: '{task}' às {target_time}")
        return {"message": {"content": f"Combinado, às {target_time} lembro-te: {task}."}}

    system_instruction = None
    history_parts = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            system_instruction = content
        elif role == "user":
            history_parts.append(types.Content(role="user", parts=[types.Part.from_text(text=content)]))
        elif role == "assistant":
            history_parts.append(types.Content(role="model", parts=[types.Part.from_text(text=content)]))

    try:
        now = datetime.now(ZoneInfo("Europe/Lisbon"))
        time_context = (
            f"\n\nContexto actual: agora são {now.strftime('%H:%M')} "
            f"do dia {now.strftime('%d/%m/%Y')}, em Lisboa, Portugal. "
            "Se te perguntarem as horas, a data, ou o dia da semana, responde sempre com este valor real -- "
            "nunca digas que não sabes nem inventes uma desculpa."
        )
    except Exception as e:
        print(f"[Aviso] Falha ao obter hora de Lisboa: {e}")
        time_context = ""

    if memory_facts:
        memory_context = "\n\nCoisas que já sabes sobre a pessoa com quem estás a falar (usa isto naturalmente, sem as recitar todas de uma vez nem as mencionar explicitamente que estás a 'consultar'):\n"
        memory_context += "\n".join(f"- {fact}" for fact in memory_facts)
    else:
        memory_context = ""

    full_system_instruction = (system_instruction or "") + time_context + memory_context

    try:
        response = gemini_client.models.generate_content(
            model=requested_model,
            contents=history_parts,
            config=types.GenerateContentConfig(
                system_instruction=full_system_instruction,
                max_output_tokens=800,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW,
                ),
            ),
        )
        reply_text = (response.text or "").strip()
    except Exception as e:
        print(f"[Gemini ERRO]: {e}")
        reply_text = "Desculpa, tive um problema a ligar ao Gemini."

    return {"message": {"content": reply_text}}


@app.get("/reminders/due", dependencies=[Depends(verify_secret)])
async def reminders_due(time: str):
    due = [r for r in reminders if not r["delivered"] and r["time"] == time]
    for r in due:
        r["delivered"] = True
    if due:
        save_reminders(reminders)
    return {"reminders": [r["task"] for r in due]}
