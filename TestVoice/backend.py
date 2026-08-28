from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from groq import Groq
import edge_tts
import tempfile
import uuid
import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODELS ----------------
asr_model = WhisperModel("small", device="cpu", compute_type="int8")
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- LLM ----------------
def ask_llm(text: str, user_lang:str):
    res = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[ 
            {
            "role": "system",
            "content": f"""
You are a multilingual assistant.

RULES:
- Respond ONLY in {user_lang}
- Do NOT translate to English unless user speaks English
- If user speaks Hindi → reply in Hindi
- If Marathi → reply in Marathi
- If Kannada → reply in Kannada
- Keep meaning natural, not translated English style

Be concise.
"""
        },
    
            {"role": "user", "content": text}
        ]
    )
    return res.choices[0].message.content


# ---------------- ASR ----------------
def transcribe(path: str, lang: str):
    segments, _ = asr_model.transcribe(path, language=lang)
    return " ".join([s.text for s in segments])


# ---------------- TTS ----------------
async def text_to_speech(text: str, voice: str):
    file_path = f"{uuid.uuid4()}.mp3"

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice
    )

    await communicate.save(file_path)
    return file_path


def detect_voice(text: str):
    # Kannada
    if any("\u0C80" <= c <= "\u0CFF" for c in text):
        return "kn-IN-SapnaNeural"

    # Hindi / Marathi
    if any("\u0900" <= c <= "\u097F" for c in text):
        return "hi-IN-SwaraNeural"

    return "en-US-AriaNeural"


# ---------------- MAIN API ----------------
@app.post("/chat")
async def chat(file: UploadFile = File(...), lang: str = Form(...)):

    try:
        # save audio
        suffix = os.path.splitext(file.filename)[-1] or ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            audio_path = tmp.name

        # ASR
        user_text = transcribe(audio_path, lang)
        print("USER:", user_text)

        # LLM
        answer = ask_llm(user_text, lang)
        print("BOT:", answer)

        # TTS
        voice = detect_voice(answer)
        audio_file = await text_to_speech(answer, voice)

        os.remove(audio_path)

        return FileResponse(
            audio_file,
            media_type="audio/mpeg",
            filename="response.mp3"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))