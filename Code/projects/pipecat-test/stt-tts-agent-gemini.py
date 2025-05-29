#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
import argparse

import aiohttp
import torch
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.services.google import GoogleTTSService
from google.ai.generativelanguage_v1beta.types import Content, Part
from pipecat.services.whisper import WhisperSTTService, Model
from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper

# Import depuis googlecode.py
from googlecode import (
    GoogleLLMService,
    GoogleLLMContext,
)

# Charger les informations d'identification
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = #...
os.environ["DAILY_API_KEY"] = #...
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# -----------------------------------------------------------------------------
# Configuration des paramètres de la session Daily
# -----------------------------------------------------------------------------

async def configure(aiohttp_session: aiohttp.ClientSession):
    parser = argparse.ArgumentParser(description="Daily AI SDK Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=False, help="URL of the Daily room to join"
    )
    parser.add_argument(
        "-k",
        "--apikey",
        type=str,
        required=False,
        help="Daily API Key (needed to create an owner token for the room)",
    )

    args, unknown = parser.parse_known_args()

    url = args.url or os.getenv("DAILY_SAMPLE_ROOM_URL")
    key = args.apikey or os.getenv("DAILY_API_KEY")

    if not url:
        raise Exception(
            "No Daily room specified. use the -u/--url option from the command line, or set DAILY_SAMPLE_ROOM_URL in your environment to specify a Daily room URL."
        )

    if not key:
        raise Exception(
            "No Daily API key specified. use the -k/--apikey option from the command line, or set DAILY_API_KEY in your environment to specify a Daily API key, available from https://dashboard.daily.co/developers."
        )

    daily_rest_helper = DailyRESTHelper(
        daily_api_key=key,
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )

    # Create a meeting token for the given room with an expiration 1 hour in the future.
    expiry_time: float = 60 * 60

    token = await daily_rest_helper.get_token(url, expiry_time)

    return (url, token)

# -----------------------------------------------------------------------------
# Initialisation du transport Daily
# -----------------------------------------------------------------------------

def initialize_transport(room_url, token):
    return DailyTransport(
        room_url,
        token,
        "Respond bot",
        DailyParams(
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

# -----------------------------------------------------------------------------
# Initialisation du service de transcription vocale
# -----------------------------------------------------------------------------

def initialize_stt():
    return WhisperSTTService(
        model=Model.BASE,
        device="cpu",
        no_speech_prob=0.4
    )

# -----------------------------------------------------------------------------
# Initialisation du service TTS de Google
# -----------------------------------------------------------------------------

def initialize_tts():
    return GoogleTTSService(
        voice_id="en-US-Neural2-C",
        params=GoogleTTSService.InputParams(language="en", rate="1"),
    )

# -----------------------------------------------------------------------------
# Initialisation du service LLM de Google
# -----------------------------------------------------------------------------

def initialize_llm():
    return GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-1.5-flash-8b")

# -----------------------------------------------------------------------------
# Initialisation du contexte et de l'agrégateur de contexte pour LLM Google
# -----------------------------------------------------------------------------

def initialize_context_and_aggregator():
    google_messages = [
        Content(
            role="assistant",
            parts=[Part(text="You are a helpful LLM in a call as a medical appointment assistant. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a factual and easily understandable way.")]
        )
    ]
    context = GoogleLLMContext(messages=google_messages)
    context_aggregator = GoogleLLMService.create_context_aggregator(context)
    return context, context_aggregator

# -----------------------------------------------------------------------------
# Initialisation du pipeline
# -----------------------------------------------------------------------------

def initialize_pipeline(transport, stt, context_aggregator, llm, tts):
    return Pipeline(
        [
            transport.input(),          # Entrée de l'utilisateur
            stt,                         # Utiliser Whisper pour la transcription
            context_aggregator.user(),   # Agrégateur de contexte utilisateur
            llm,                         # Modèle LLM Google
            tts,                         # Service TTS
            transport.output(),          # Sortie audio vers l'utilisateur
            context_aggregator.assistant()  # Agrégateur de contexte assistant
        ]
    )

# -----------------------------------------------------------------------------
# Initialisation de la tâche du pipeline
# -----------------------------------------------------------------------------

def initialize_task(pipeline):
    return PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

# -----------------------------------------------------------------------------
# Gestion de l'événement à la première connexion d'un participant
# -----------------------------------------------------------------------------

def handle_first_participant_joined(transport, task, google_messages):
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        transport.capture_participant_transcription(participant["id"])
        google_messages.append(Content(
            role="assistant",
            parts=[Part(text="Please introduce yourself to the user as a medical appointment assistant.")]
        ))
        await task.queue_frames([LLMMessagesFrame(google_messages)])

# -----------------------------------------------------------------------------
# Fonction principale pour initialiser et démarrer le pipeline
# -----------------------------------------------------------------------------

async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = initialize_transport(room_url, token)
        stt = initialize_stt()
        tts = initialize_tts()
        llm = initialize_llm()
        context, context_aggregator = initialize_context_and_aggregator()

        pipeline = initialize_pipeline(transport, stt, context_aggregator, llm, tts)
        task = initialize_task(pipeline)

        handle_first_participant_joined(transport, task, context.messages)

        # Démarrage du pipeline
        runner = PipelineRunner()
        await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())
