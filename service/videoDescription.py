import os
import torch
import cv2
import random
import pandas as pd
import moviepy.editor as mp
import whisper
import shutil
import numpy as np
import base64
import requests
from resemblyzer import VoiceEncoder, preprocess_wav
from pydub import AudioSegment
from scipy.spatial.distance import cosine
from pipeline_model.utils.model_loader import load_clip
from pydub.utils import mediainfo
from spectralcluster import SpectralClusterer

device = "cuda" if torch.cuda.is_available() else "cpu"

# carregnado modelo de Clip embeddings
clip_model, clip_processor = load_clip()

# Lista de emoções
emotions = ["feliz", "neutro", "triste", "raiva", "medo", "surpreso", "calmo", "tédio", "cansado", "irritado"]

def generate_correlation_matrix(synced_data):
    if not synced_data:
        raise ValueError("Erro: synced_data está vazio ou None!")

    # Verifica se synced_data é uma lista de tuplas do tamanho correto
    if not isinstance(synced_data, list) or not all(
        isinstance(row, tuple) and len(row) == 4 for row in synced_data
    ):
        raise ValueError(f"Formato inválido de synced_data: {synced_data}")

    return pd.DataFrame(
        synced_data,
        columns=[
            "Timestamp",
            "Frame",
            "Transcription",
            "OCR_Text",
        ],
    )


# Função para extração de texto (OCR)
def extract_text(image_path, reader):
    """
    Extrai texto de uma imagem usando OCR.
    
    :param image_path: Caminho da imagem a ser processada.
    :param reader: Modelo OCR para leitura de texto.
    :return: Texto extraído da imagem.
    """
    results = reader.readtext(image_path, detail=0)
    return " ".join(results) if results else ""

# Função para pré-processamento da imagem
def preprocess_image(image_path):
    """
    Converte uma imagem para escala de cinza e aplica equalização de histograma.
    
    :param image_path: Caminho da imagem.
    :return: Imagem processada em escala de cinza com limiarização adaptativa.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C_,
                                      cv2.THRESH_BINARY, 11, 2)
    return processed


# Função para detectar emoção no áudio
def detect_emotion_from_audio(audio_array, sr, audio_processor, audio_model):
    """Detecta a emoção diretamente de um array de áudio sem salvar no disco.
    
    :param audio_array: Array numpy contendo os dados de áudio.
    :param sr: Taxa de amostragem do áudio.
    :param audio_processor: Processador de áudio para transformar os dados.
    :param audio_model: Modelo de IA para detecção de emoções.
    :return: Emoção detectada e sua confiança.
    """
    input_values = audio_processor(audio_array, sampling_rate=sr, return_tensors="pt", padding=True).input_values.to(device)
    with torch.no_grad():
        logits = audio_model(input_values).logits
    scores = torch.nn.functional.softmax(logits, dim=-1)
    emotion_idx = scores.argmax().item()
    emotion = emotions[emotion_idx]
    confidence = scores[0, emotion_idx].item()
    return emotion, confidence


def transcribe_video_with_timestamps(video_path, model_type="large"):
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError("FFmpeg não foi encontrado. Instale e adicione ao PATH.")

    # 1. Extrair áudio do vídeo
    audio_path = "temp_audio.wav"
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)

    info = mediainfo(audio_path)
    print(f"Duração detectada do áudio extraído: {info['duration']} segundos")

    # 2. Transcrever com Whisper
    model = whisper.load_model(model_type)
    result = model.transcribe(
        audio_path,
        fp16=False,
        word_timestamps=True,
        temperature=0.0,
        initial_prompt="",
        no_speech_threshold=0.0,
    )
    transcription_segments = result["segments"]

    # 3. Carregar o áudio completo com pydub para segmentação
    audio = AudioSegment.from_wav(audio_path)

    # 4. Extrair embeddings por segmento de transcrição
    encoder = VoiceEncoder()
    embeddings = []
    valid_segments = []

    for seg in transcription_segments:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        segment_audio = audio[start_ms:end_ms]
        temp_seg_path = "temp_seg.wav"
        segment_audio.export(temp_seg_path, format="wav")

        try:
            wav = preprocess_wav(temp_seg_path)
            embed = encoder.embed_utterance(wav)
            embeddings.append(embed)
            valid_segments.append(seg)
        except Exception as e:
            continue

        os.remove(temp_seg_path)

    # 5. Clustering dos embeddings
    clusterer = SpectralClusterer(
        min_clusters=1,
        max_clusters=5,
        # autotune=AutoTune(p_percentile=0.90, gaussian_blur_sigma=1.0) #Possível otimização futura
    )
    labels = clusterer.predict(np.stack(embeddings))

    # 6. Associar labels de locutor à transcrição
    annotated_segments = []
    for seg, label in zip(valid_segments, labels):
        annotated_segments.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "speaker": f"Speaker_{label}",
            }
        )

    os.remove(audio_path)
    return annotated_segments, result


def get_clip_embedding(frame):
    inputs = clip_processor(images=[frame], return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    return embedding.cpu().numpy().flatten()


def summarize_video_frames(video_path, frame_interval=1, threshold=0.2, window_size=5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    current_frame_index = 0
    saved_frames = []

    # Carrega o primeiro frame como referência
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
    ret, reference_frame = cap.read()
    if not ret:
        cap.release()
        return []

    reference_embedding = get_clip_embedding(reference_frame)
    saved_frames.append((0.0, reference_frame))

    while current_frame_index < total_frames:
        change_detected = False

        for offset in range(1, window_size + 1):
            next_index = current_frame_index + offset * int(fps * frame_interval)
            if next_index >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, next_index)
            ret, next_frame = cap.read()
            if not ret:
                continue

            next_embedding = get_clip_embedding(next_frame)
            distance = cosine(reference_embedding, next_embedding)

            if distance > threshold:
                timestamp = next_index / fps
                saved_frames.append((timestamp, next_frame))
                reference_embedding = next_embedding
                current_frame_index = next_index
                change_detected = True
                break

        if not change_detected:
            current_frame_index += window_size * int(fps * frame_interval)

    cap.release()
    return saved_frames


def save_summarized_frames(
    video_path,
    output_folder="frames_summary",
    frame_interval=1,
    threshold=0.25,
    window_size=5,
):
    os.makedirs(output_folder, exist_ok=True)
    summarized_frames = summarize_video_frames(
        video_path, frame_interval, threshold, window_size
    )

    saved_paths = []
    for timestamp, frame in summarized_frames:
        frame_path = os.path.join(output_folder, f"scene_{timestamp:.2f}.jpg")
        cv2.imwrite(frame_path, frame)
        saved_paths.append((timestamp, frame_path))

    return saved_paths


def describe_image(
    image_path,
    model="llama3.2-vision:11b",
    api_host="",
):
    import json

    def parse_multiple_json_responses(text_response):
        responses = []
        for line in text_response.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if "response" in parsed:
                    responses.append(parsed["response"])
            except json.JSONDecodeError:
                continue
        # Junta os pedaços e remove quebras de linha artificiais
        joined = "".join(responses)
        return joined.replace("\\n", " ").replace("\\", "").strip()

    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

    payload = {
        "model": model,
        "prompt": "Descreva de forma breve e objetiva o que está visível na imagem, em português.",
        "images": [encoded_image],
    }

    response = requests.post(f"{api_host}/api/generate", json=payload)

    if response.status_code == 200:
        result = parse_multiple_json_responses(response.text)
        return result
    else:
        raise RuntimeError(
            f"Erro na requisição à API do Ollama: {response.status_code} - {response.text}"
        )


# Função para sincronizar áudio, frames e OCR
def sync_audio_and_frames(
    transcription_segments,
    frame_list,
    audio_path,
    reader,
    translator,
    image_processor,
    image_model,
    audio_processor,
    audio_model,
):
    """
    Sincroniza frames com transcrição de áudio e OCR por janelas entre dois frames.

    :param transcription_segments: Lista de dicionários com 'start', 'end', 'text', 'speaker'
    :param frame_list: Lista de tuplas (timestamp, frame_path)
    :param audio_path: Caminho para o vídeo (utilizado para validar duração)
    :param reader: Leitor OCR
    :return: Lista de tuplas: (timestamp_inicial, frame_path, transcription_text, description, emotion, confidence, ocr_text)
    """
    synced_data = []
    video = mp.VideoFileClip(audio_path)
    video_duration = video.duration

    num_frames = len(frame_list)

    for idx, (start_time, frame_path) in enumerate(frame_list):
        end_time = frame_list[idx + 1][0] if idx + 1 < num_frames else video_duration

        selected_segments = [
            seg
            for seg in transcription_segments
            if not (seg["end"] <= start_time or seg["start"] >= end_time)
        ]

        full_text = ""
        last_speaker = None

        for seg in selected_segments:
            current_speaker = seg.get("speaker", "Unknown")
            if current_speaker != last_speaker:
                if full_text:
                    full_text += "\n"
                full_text += f"{current_speaker}: {seg['text']}"
            else:
                full_text += f" {seg['text']}"
            last_speaker = current_speaker

        # ocr_text = extract_text(frame_path, reader)
        ocr_text = describe_image(frame_path)
        synced_data.append((start_time, frame_path, full_text.strip(), ocr_text))

    return synced_data
