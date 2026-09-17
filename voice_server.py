import os
import re
import io
import json
import uuid
import tempfile
import subprocess
import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from datetime import datetime
from zoneinfo import ZoneInfo
import base64
import time
import requests
from fastapi.responses import HTMLResponse, RedirectResponse

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
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")
WEATHER_LAT = 38.7223
WEATHER_LON = -9.1393

def verify_secret(x_pickle_secret: str = Header(default="")):
    if not PICKLE_SHARED_SECRET or x_pickle_secret != PICKLE_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Não autorizado")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

GEMINI_MODEL = "gemini-3.1-flash-lite"
GROQ_STT_MODEL = "whisper-large-v3-turbo"

# ==========================================
# Configuração Spotify
# ==========================================
SPOTIFY_CLIENT_ID     = os.environ["CLIENT_ID_SPOTIFY"]
SPOTIFY_CLIENT_SECRET = os.environ["CLIENT_SECRET_SPOTIFY"]
SPOTIFY_REDIRECT_URI  = "https://esp32s3-touch-lcd-3-5-pickle-robot.onrender.com/callback"
SPOTIFY_SCOPES        = "user-read-currently-playing user-read-playback-state user-modify-playback-state"

SPOTIFY_TOKENS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spotify_tokens.json")

def load_spotify_tokens() -> dict:
    if not os.path.exists(SPOTIFY_TOKENS_FILE): return {}
    try:
        with open(SPOTIFY_TOKENS_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return {}

def save_spotify_tokens(tokens: dict):
    with open(SPOTIFY_TOKENS_FILE, "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)

spotify_tokens = load_spotify_tokens()

def get_spotify_access_token():
    if not spotify_tokens.get("refresh_token"):
        return None
    if spotify_tokens.get("access_token") and time.time() < spotify_tokens.get("expires_at", 0):
        return spotify_tokens["access_token"]

    auth_header = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={"Authorization": f"Basic {auth_header}", "Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "refresh_token", "refresh_token": spotify_tokens["refresh_token"]},
    )
    if resp.status_code != 200:
        print(f"[Spotify] Erro ao renovar token: {resp.text}")
        return None

    data = resp.json()
    spotify_tokens["access_token"] = data["access_token"]
    spotify_tokens["expires_at"]   = time.time() + data["expires_in"] - 30
    if "refresh_token" in data:
        spotify_tokens["refresh_token"] = data["refresh_token"]
    save_spotify_tokens(spotify_tokens)
    return spotify_tokens["access_token"]

_tempo_cache = {"track_id": None, "tempo": 0.0}


DEFAULT_TEMPO_BPM = 100.0

def get_track_tempo(track_id: str, access_token: str) -> float:
    if _tempo_cache["track_id"] == track_id:
        return _tempo_cache["tempo"]

    resp = requests.get(
        f"https://api.spotify.com/v1/audio-features/{track_id}",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    if resp.status_code == 200:
        tempo = resp.json().get("tempo", 0.0) or DEFAULT_TEMPO_BPM
    else:
        print(f"[Spotify] audio-features falhou ({resp.status_code}): {resp.text[:200]}")
        tempo = DEFAULT_TEMPO_BPM

    _tempo_cache["track_id"] = track_id
    _tempo_cache["tempo"] = tempo
    return tempo

ACCEPTED_VARIANTS = [
    "pickle", "picle", "pico", "pika", "pica", 
    "pekle", "pikl", "becle", "piclo", "pizzel"
]

WHISPER_SILENCE_HALLUCINATIONS = {
    "obrigado", "obrigada", "obrigado.", "obrigada.",
    "subscreva", "inscreva-se", "deixe o seu like",
    "amém", "amém.", "obrigado por assistir", "legendas:",
    "obrigado pela vossa atenção", "já está", "tchau",
    "o que podes fazer", "queso"
}

COMMON_PHRASE_CORRECTIONS = {
    r'\bponto\s+chumas\b': 'como te chamas',
    r'\bvamos\s+juntos\s*,\s*chamas\b': 'como te chamas',
    r'\bchumas\b': 'chamas',
    r'\bcom\s+te\s+chamas\b': 'como te chamas',
    r'\bquem\s+es\s+tu\b': 'quem és tu',
    r'\bque\s+e\s+isso\b': 'o que é isso',
    r'\bessa\s+viol[êe]ncia\b': 'isso é violência'
}

def fix_stt_phonetics(text: str) -> str:
    corrected = text
    for pattern, replacement in COMMON_PHRASE_CORRECTIONS.items():
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
    return corrected

# [ALTERAÇÃO] VAD ajustado para ser permissivo com sinais fracos do ESP32
def contains_speech_vad(audio_bytes: bytes) -> bool:
    try:
        import webrtcvad
        # Mudado de 3 (muito rigoroso) para 1 (permissivo para mics de longe/baixo volume)
        vad = webrtcvad.Vad(2)
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
        # Reduzido de 0.15 (15%) para 0.05 (5%) para aceitar frases mais suaves
        return (speech_frames / total_frames) > 0.10
    except Exception:
        return True

MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickle_memory.json")
REMINDERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickle_reminders.json")

def load_reminders():
    if not os.path.exists(REMINDERS_FILE): return []
    try:
        with open(REMINDERS_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return []

def save_reminders(reminders_list):
    with open(REMINDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(reminders_list, f, ensure_ascii=False, indent=2)

def load_memory() -> list:
    if not os.path.exists(MEMORY_FILE): return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return []

def save_memory(facts: list):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(facts, f, ensure_ascii=False, indent=2)

reminders = load_reminders()
memory_facts = load_memory()

def clean_portuguese_text(text: str) -> str:
    if re.search(r'[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\u4e00-\u9faf]', text): return ""
    return re.sub(r'[^a-zA-Z0-9áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ\s\.,!\?\'-]', '', text).strip()

def is_hallucination(text: str) -> bool:
    words = text.lower().split()
    if not words: return False
    current_repeat = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            current_repeat += 1
            if current_repeat >= 3: return True
        else: current_repeat = 1
    if len(words) >= 3 and len(set(words)) == 1: return True
    clean_text = " ".join(words)
    return bool(re.search(r'(\b\w+\s+\w+\b)(?:\s+\1){2,}', clean_text))

NO_SPEECH_PROB_THRESHOLD = 0.6
AVG_LOGPROB_REJECT_THRESHOLD = -1.0
AVG_LOGPROB_SOFT_THRESHOLD = -0.5
COMPRESSION_RATIO_THRESHOLD = 2.4

def _extract_confident_text(transcription) -> str:
    """
    Usa os metadados por segmento (no_speech_prob, avg_logprob, compression_ratio)
    do response_format=verbose_json para descartar segmentos que o Whisper
    alucinou a partir de silêncio/ruído -- em vez de confiar cegamente em
    transcription.text.
    """
    segments = getattr(transcription, "segments", None)
    if not segments:
        # Sem segmentos (fallback) -- mantém o comportamento anterior.
        return (getattr(transcription, "text", "") or "").strip()

    kept_parts = []
    for seg in segments:
        get = (lambda k, d=None: seg.get(k, d)) if isinstance(seg, dict) else (lambda k, d=None: getattr(seg, k, d))
        no_speech_prob = get("no_speech_prob", 0.0)
        avg_logprob = get("avg_logprob", 0.0)
        compression_ratio = get("compression_ratio", 1.0)
        text = get("text", "")

        if avg_logprob is not None and avg_logprob < AVG_LOGPROB_REJECT_THRESHOLD:
            print(f"[STT] Segmento rejeitado (avg_logprob={avg_logprob:.2f}): '{text}'")
            continue
        if (no_speech_prob is not None and no_speech_prob > NO_SPEECH_PROB_THRESHOLD
                and avg_logprob is not None and avg_logprob < AVG_LOGPROB_SOFT_THRESHOLD):
            print(f"[STT] Segmento rejeitado (no_speech_prob={no_speech_prob:.2f}, avg_logprob={avg_logprob:.2f}): '{text}'")
            continue
        if compression_ratio is not None and compression_ratio > COMPRESSION_RATIO_THRESHOLD:
            print(f"[STT] Segmento rejeitado (compression_ratio={compression_ratio:.2f}): '{text}'")
            continue

        kept_parts.append(text)

    return " ".join(p.strip() for p in kept_parts if p and p.strip())

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

@app.post("/stt", dependencies=[Depends(verify_secret)])
async def stt(request: Request):
    audio_bytes = await request.body()
    
    if not contains_speech_vad(audio_bytes):
        print("[STT Ignorado]: Ruído de fundo sem presença de voz")
        return {"text": ""}

    try:
        transcription = groq_client.audio.transcriptions.create(
            file=("audio.wav", io.BytesIO(audio_bytes), "audio/wav"),
            model=GROQ_STT_MODEL,
            prompt="Transcrição em português de Portugal para o robô Pickle. Perguntas e exclamações comuns: Como te chamas?, Olá Pickle, Quem és tu?, Isso é violência!, Que horas são?, O que podes fazer?.",
            response_format="verbose_json",
            language="pt",
            temperature=0.0
        )
    except Exception as e:
        print(f"[Groq STT ERRO]: {e}")
        return {"text": ""}

    raw_text = _extract_confident_text(transcription)
    if not raw_text:
        print("[STT Ignorado]: segmentos rejeitados por baixa confiança (provável alucinação em silêncio)")
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
        return {"text": ""}

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
        await asyncio.wait_for(communicate.save(tmp_mp3), timeout=10)

        # subprocess.run é bloqueante -- corre numa thread separada para não
        # congelar o event loop (e, com ele, /chat, /stt, /spotify/*, etc.)
        await asyncio.to_thread(
            subprocess.run,
            ["ffmpeg", "-y", "-i", tmp_mp3, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", tmp_wav],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10
        )

        with open(tmp_wav, "rb") as f:
            wav_data = f.read()

        return Response(content=wav_data, media_type="audio/wav")
    except (asyncio.TimeoutError, subprocess.TimeoutExpired):
        print("[Edge-TTS ERRO]: timeout a gerar áudio (TTS lento demais, abortado)")
        return Response(content=b"", media_type="audio/wav", status_code=504)
    except Exception as e:
        print(f"[Edge-TTS ERRO]: {e}")
        return Response(content=b"", media_type="audio/wav", status_code=500)
    finally:
        if os.path.exists(tmp_mp3): os.remove(tmp_mp3)
        if os.path.exists(tmp_wav): os.remove(tmp_wav)

# ==========================================
# Meteorologia (OpenWeather)
# ==========================================
def _detect_weather_query(text: str):
    """Devolve 'tomorrow', 'today' ou None consoante o pedido de meteorologia."""
    lower = text.lower()
    is_weather_query = any(kw in lower for kw in [
        "previsão", "previsao", "meteorologia", "meteorológico", "meteorologico",
        "que tempo faz", "como está o tempo", "como esta o tempo", "vai chover",
        "vai estar sol", "faz frio", "faz calor",
    ])
    if not is_weather_query:
        return None
    if "amanhã" in lower or "amanha" in lower:
        return "tomorrow"
    return "today"

def get_current_weather() -> str:
    if not WEATHER_API_KEY:
        return "Não tenho acesso à API do tempo neste momento."
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "lat": WEATHER_LAT, "lon": WEATHER_LON,
                "appid": WEATHER_API_KEY, "units": "metric", "lang": "pt",
            },
            timeout=6,
        )
        resp.raise_for_status()
        data = resp.json()
        temp = round(data["main"]["temp"])
        feels_like = round(data["main"]["feels_like"])
        description = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        return (
            f"Agora em Lisboa estão {temp} graus, com {description}. "
            f"Sensação térmica de {feels_like} graus e {humidity} por cento de humidade."
        )
    except Exception as e:
        print(f"[Weather ERRO]: {e}")
        return "Não consegui obter a meteorologia atual, desculpa."

def get_daily_forecast(days_ahead: int = 0) -> str:
    """days_ahead: 0 = hoje, 1 = amanhã. Agrega o endpoint /forecast (passos de 3h) para o dia alvo."""
    if not WEATHER_API_KEY:
        return "Não tenho acesso à API do tempo neste momento."
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={
                "lat": WEATHER_LAT, "lon": WEATHER_LON,
                "appid": WEATHER_API_KEY, "units": "metric", "lang": "pt",
            },
            timeout=6,
        )
        resp.raise_for_status()
        data = resp.json()

        target_date = (datetime.now(ZoneInfo("Europe/Lisbon")) + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        day_entries = [e for e in data.get("list", []) if e["dt_txt"].startswith(target_date)]
        if not day_entries:
            return "Ainda não tenho previsão suficiente para esse dia."

        temps = [e["main"]["temp"] for e in day_entries]
        min_temp = round(min(temps))
        max_temp = round(max(temps))
        midday_entry = min(day_entries, key=lambda e: abs(int(e["dt_txt"][11:13]) - 13))
        description = midday_entry["weather"][0]["description"]

        day_label = "hoje" if days_ahead == 0 else "amanhã"
        return (
            f"A previsão para {day_label} em Lisboa aponta para {description}, "
            f"com temperaturas entre os {min_temp} e os {max_temp} graus."
        )
    except Exception as e:
        print(f"[Weather ERRO]: {e}")
        return "Não consegui obter a previsão do tempo, desculpa."

# ==========================================
# Briefing matinal (Meteorologia + Notícias)
# ==========================================
NEWS_FEEDS = [
    ("RTP", "https://www.rtp.pt/noticias/rss"),
    ("Público", "https://feeds.feedburner.com/PublicoRSS"),
]

def fetch_news_headlines(per_feed: int = 2) -> list:
    """Vai buscar as manchetes mais recentes de cada fonte RSS.
    Cada fonte falha isoladamente -- uma fonte em baixo não deve
    destruir o briefing inteiro."""
    headlines = []
    for source_name, feed_url in NEWS_FEEDS:
        try:
            resp = requests.get(feed_url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            items = root.findall("./channel/item")[:per_feed]
            for item in items:
                title_el = item.find("title")
                if title_el is not None and title_el.text:
                    headlines.append((source_name, title_el.text.strip()))
        except Exception as e:
            print(f"[News ERRO] Falhou obter feed de {source_name}: {e}")
    return headlines

def get_morning_briefing() -> str:
    weather_line = get_current_weather()
    headlines = fetch_news_headlines(per_feed=2)

    if not headlines:
        return f"Bom dia! {weather_line} Não consegui ir buscar as notícias agora, tenta mais tarde."

    news_parts = [f"Da {source}: {title}." for source, title in headlines]
    news_block = " ".join(news_parts)

    return f"Bom dia! {weather_line} Agora as principais notícias. {news_block}"


@app.post("/chat", dependencies=[Depends(verify_secret)])
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    last_user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_message = msg.get("content", "").strip()
            break

    lower_msg = last_user_message.lower()

    remember_match = re.match(r'^lembra[\s\-,]*te[\s,]*(?:que\s+)?', lower_msg)
    if remember_match:
        new_fact = last_user_message[remember_match.end():].strip(" ,.")
        if new_fact:
            memory_facts.append(new_fact)
            save_memory(memory_facts)
            return {"message": {"content": f"Ok, vou lembrar-me disso: {new_fact}."}}

    if re.match(r'^esquece\s+tudo', lower_msg):
        memory_facts.clear()
        save_memory(memory_facts)
        return {"message": {"content": "Pronto, esqueci tudo o que sabia sobre ti."}}


    if re.match(r'^bom\s*dia\b', lower_msg):
        return {"message": {"content": get_morning_briefing()}}

        
    weather_query = _detect_weather_query(last_user_message)
    if weather_query == "tomorrow":
        return {"message": {"content": get_daily_forecast(days_ahead=1)}}
    elif weather_query == "today":
        if "previsão" in lower_msg or "previsao" in lower_msg:
            return {"message": {"content": get_daily_forecast(days_ahead=0)}}
        return {"message": {"content": get_current_weather()}}

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
            "Se te perguntarem as horas, a data, ou o dia da semana, responde sempre com este valor real."
        )
    except Exception:
        time_context = ""

    memory_context = "\n\nCoisas que já sabes sobre a pessoa:\n" + "\n".join(f"- {fact}" for fact in memory_facts) if memory_facts else ""
    full_system_instruction = (system_instruction or "") + time_context + memory_context

    try:
        response = gemini_client.models.generate_content(
            model=requested_model,
            contents=history_parts,
            config=types.GenerateContentConfig(
                system_instruction=full_system_instruction,
                max_output_tokens=800,
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            ),
        )
        reply_text = (response.text or "").strip()
    except Exception as e:
        print(f"[Gemini ERRO]: {e}")
        reply_text = "Desculpa, tive um problema a ligar ao Gemini."

    return {"message": {"content": reply_text}}

@app.get("/spotify/login")
async def spotify_login():
    from urllib.parse import urlencode
    params = {
        "client_id": SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "scope": SPOTIFY_SCOPES,
    }
    return RedirectResponse("https://accounts.spotify.com/authorize?" + urlencode(params))

@app.get("/callback")
async def spotify_callback(code: str = None, error: str = None):
    if error:
        return HTMLResponse(f"<h3>Erro na autorização do Spotify: {error}</h3>")
    if not code:
        return HTMLResponse("<h3>Pedido inválido (sem código).</h3>")

    auth_header = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={"Authorization": f"Basic {auth_header}", "Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "authorization_code", "code": code, "redirect_uri": SPOTIFY_REDIRECT_URI},
    )
    if resp.status_code != 200:
        return HTMLResponse(f"<h3>Falha ao trocar o código por token: {resp.text}</h3>")

    data = resp.json()
    spotify_tokens["access_token"]  = data["access_token"]
    spotify_tokens["refresh_token"] = data["refresh_token"]
    spotify_tokens["expires_at"]    = time.time() + data["expires_in"] - 30
    save_spotify_tokens(spotify_tokens)

    return HTMLResponse("<h3>Spotify autorizado! Já podes fechar esta janela.</h3>")

@app.get("/spotify/now-playing", dependencies=[Depends(verify_secret)])
async def spotify_now_playing():
    access_token = get_spotify_access_token()
    if not access_token:
        return {"playing": False}

    resp = requests.get(
        "https://api.spotify.com/v1/me/player/currently-playing",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    if resp.status_code != 200:
        return {"playing": False}

    data = resp.json()
    item = data.get("item")
    if not item or not data.get("is_playing"):
        return {"playing": False}

    track   = item.get("name", "")
    artists = ", ".join(a.get("name", "") for a in item.get("artists", []))
    tempo   = get_track_tempo(item.get("id", ""), access_token) if item.get("id") else 0.0

    return {"playing": True, "track": track, "artist": artists, "tempo": round(tempo, 1)}

@app.post("/spotify/play", dependencies=[Depends(verify_secret)])
async def spotify_play():
    access_token = get_spotify_access_token()
    if not access_token: raise HTTPException(status_code=401, detail="Spotify não autorizado")
    requests.put("https://api.spotify.com/v1/me/player/play", headers={"Authorization": f"Bearer {access_token}"})
    return {"ok": True}

@app.post("/spotify/pause", dependencies=[Depends(verify_secret)])
async def spotify_pause():
    access_token = get_spotify_access_token()
    if not access_token: raise HTTPException(status_code=401, detail="Spotify não autorizado")
    requests.put("https://api.spotify.com/v1/me/player/pause", headers={"Authorization": f"Bearer {access_token}"})
    return {"ok": True}

@app.post("/spotify/next", dependencies=[Depends(verify_secret)])
async def spotify_next():
    access_token = get_spotify_access_token()
    if not access_token: raise HTTPException(status_code=401, detail="Spotify não autorizado")
    requests.post("https://api.spotify.com/v1/me/player/next", headers={"Authorization": f"Bearer {access_token}"})
    return {"ok": True}

@app.post("/spotify/toggle-play", dependencies=[Depends(verify_secret)])
async def spotify_toggle_play():
    access_token = get_spotify_access_token()
    if not access_token: raise HTTPException(status_code=401, detail="Spotify não autorizado")

    resp = requests.get("https://api.spotify.com/v1/me/player", headers={"Authorization": f"Bearer {access_token}"})
    is_playing = resp.json().get("is_playing", False) if (resp.status_code == 200 and resp.text) else False

    action = "pause" if is_playing else "play"
    requests.put(f"https://api.spotify.com/v1/me/player/{action}", headers={"Authorization": f"Bearer {access_token}"})
    return {"ok": True, "action": action}

@app.post("/spotify/previous", dependencies=[Depends(verify_secret)])
async def spotify_previous():
    access_token = get_spotify_access_token()
    if not access_token: raise HTTPException(status_code=401, detail="Spotify não autorizado")
    requests.post("https://api.spotify.com/v1/me/player/previous", headers={"Authorization": f"Bearer {access_token}"})
    return {"ok": True}

@app.get("/reminders/due", dependencies=[Depends(verify_secret)])
async def reminders_due(time: str):
    due = [r for r in reminders if not r["delivered"] and r["time"] == time]
    for r in due: r["delivered"] = True
    if due: save_reminders(reminders)
    return {"reminders": [r["task"] for r in due]}
