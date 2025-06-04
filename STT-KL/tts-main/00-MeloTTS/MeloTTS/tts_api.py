from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel
from io import BytesIO
import os
import signal
import uvicorn
import numpy as np
from melo.api import TTS
import uuid

# Initialize FastAPI app
APP = FastAPI()

# Global TTS model (loaded once)
device = "cuda:0"  # or "cuda:0"
# model = TTS(language="KR", device=device, ckpt_path="/workspace/models/G_278000.pth")
model = TTS(language="KR", device=device)
speaker_ids = model.hps.data.spk2id

# Utility functions
def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str) -> BytesIO:
    """
    Packs audio data into the desired format (currently supports wav only).
    """
    if media_type == "wav":
        import soundfile as sf
        sf.write(io_buffer, data, rate, format="wav")
    else:
        raise ValueError("Unsupported media_type. Currently, only 'wav' is supported.")
    io_buffer.seek(0)
    return io_buffer

# Request Model
class TTSRequest(BaseModel):
    text: str
    speaker: str
    speed: float = 1.0
    request_id: str = None
    media_type: str = "wav"

# TTS Endpoint Handler
async def tts_handler(req: TTSRequest) -> Response:
    """
    Handles the TTS processing request.
    """
    try:
        if req.speaker not in speaker_ids:
            raise ValueError(f"Speaker {req.speaker} not found. Available speakers: {list(speaker_ids.keys())}")

        if req.request_id is None:
            req.request_id = str(uuid.uuid4())

        output_path = f"output_{req.speaker}_{req.request_id}.wav"

        # Generate TTS output
        model.tts_to_file(req.text, speaker_ids[req.speaker], output_path, speed=req.speed)

        print(output_path)
        # Load generated audio and pack into desired format
        with open(output_path, "rb") as f:
            audio_data = f.read()

        # Delete the temporary file
        os.remove(output_path)

        return Response(content=audio_data, media_type=f"audio/{req.media_type}")
    except Exception as e:
        print("Exception", str(e))
        return JSONResponse(status_code=400, content={"error": str(e)})

# API Endpoints
@APP.post("/tts")
async def tts_post(request: TTSRequest):
    """
    POST endpoint for TTS synthesis.
    """
    return await tts_handler(request)

@APP.get("/control")
async def control(command: str):
    """
    Handles control commands like restart or exit.
    """
    if command == "restart":
        os.execl(sys.executable, sys.executable, *sys.argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
    else:
        return JSONResponse(status_code=400, content={"message": f"Unknown command: {command}"})

# Main Application Runner
if __name__ == "__main__":
    uvicorn.run("tts_api:APP", host="0.0.0.0", port=9889)
