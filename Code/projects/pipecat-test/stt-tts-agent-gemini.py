#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.google import GoogleTTSService
from pipecat.services.openai import OpenAILLMService, OpenAITTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from google.ai.generativelanguage_v1beta.types import content as glm

# Import directly from your local googlecode.py
from googlecode import (
    GoogleLLMService,
    GoogleLLMContext,
    GoogleContextAggregatorPair,
    GoogleUserContextAggregator,
    GoogleAssistantContextAggregator
)

import os

# Set the path to your service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/yyymai/Code/google_service_acc_credential_API.json"

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
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

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        # tried using GoogleTTSService() but couldnt find the api on google cloud??
        tts = GoogleTTSService(
            voice_id="en-US-Neural2-C", #en-US-Neural2-J for man's voice, and fr-FR-Neural2-C for same voice in Fnrech, don't forget to change prompt into French
            params=GoogleTTSService.InputParams(language="en", rate="1"), #fr for French
        )

        llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-1.5-flash-8b") #gemini-1.5-flash-8b #gpt-4o-mini

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a call as a medical appointment assistant. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a factual and easily understandable way.",
                # change prompt to french if needed
            },
        ]

        # Create Google-formatted messages
        google_messages = [
            glm.Content(
                role="user" if msg["role"] == "system" else msg["role"],  # Google uses "user" instead of "system"
                parts=[glm.Part(text=msg["content"])]
            )
            for msg in messages
        ]

        # Create GoogleLLMContext directly
        context = GoogleLLMContext(messages=google_messages)

        # Create the context aggregator
        context_aggregator = GoogleLLMService.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,  # STT
                context_aggregator.user(),  # User respones
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            # Kick off the conversation.
            messages.append({"role": "system", "content": "Please introduce yourself to the user as a medical appointment assistant."})
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
