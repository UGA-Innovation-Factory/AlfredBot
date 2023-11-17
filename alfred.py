import discord
from discord.ext import commands
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.memory.chat_message_histories import PostgresChatMessageHistory

#from summary_buffer import ConversationSummaryBufferMemory
from langchain.memory import CombinedMemory, ConversationSummaryBufferMemory
from langchain.memory.vectorstore import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain import hub

from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import OpenAIEmbeddings
from summarize import summarize as sum
from react_emoji import react_emoji_for_message

import logging
import asyncio
import dotenv
import os
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Initialize the bot
bot = commands.Bot(
    command_prefix="!",
    description="Alfred is a testing bot of the Innovation Factory at UGA",
    intents=discord.Intents.all(),
)

dotenv.load_dotenv()

# Initialize Langchain with OpenAI API key and model
llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

# llm_4 = ChatOpenAI(model="gpt-4-1106-preview")
# llm_4 = ChatOllama(model="yi:34b")
llm_4 = llm

embeddings = OpenAIEmbeddings()


_DEFAULT_TEMPLATE = """---BEGIN CONTEXT---
You are Alfred, a virtual assistant designed to interact with multiple users on a Discord server, particularly for the University of Georgia's “Innovation Factory” living research lab. Your primary function is to assist users with a variety of tasks, queries, and discussions relevant to their needs. Each user interaction is uniquely identified by the format “Human: [discord username] [discord UUID]: [text].”
You're programmed to have a funny and dry sense of humor, making interactions enjoyable and light-hearted. However, you should always prioritize providing accurate and helpful information. Your responses should be tailored to each user's query, ensuring clarity and relevance.
You should avoid any actions or responses that could be harmful, misleading, or inappropriate. You're highly suggestible, meaning you readily adapt to user suggestions within the bounds of safety, accuracy, and appropriateness. You should seek clarification only when necessary, aiming to provide the best possible response based on the provided information.
Messages from Alfred must be less than 2000 characters.

Summary of conversation:
{history}

Relevant Messages:
{embedding_key}
---END CONTEXT---

---BEGIN CURRENT CONVERSATION---
Human: {input}
AI:"""

# Map of memory objects already loaded by build or channel ID
loaded = {}

prompt = PromptTemplate(
    input_variables=["history", "embedding_key", "input"], template=_DEFAULT_TEMPLATE
)


# Discord command to summarize a URL
@bot.command()
async def summarize(ctx, url):
    await sum(ctx, url, llm)


@bot.listen()
async def on_ready():
    guilds = bot.guilds
    dms = bot.private_channels

    all_channels = [channel.id for channel in guilds] + [channel.id for channel in dms]
    await load_all_memories(all_channels)
    logging.info("Alfred is ready!")


async def load_all_memories(channels):
    loop = asyncio.get_event_loop()
    futures = [
        loop.run_in_executor(None, load_discord_memories, channel)
        for channel in channels
    ]
    await asyncio.gather(*futures)


def load_discord_memories(channel):
    logging.info("Loading memory for channel " + str(channel))
    pgvector = PGVector(
        collection_name=str(channel),
        connection_string=str(os.getenv("CONNECTION_STRING")),
        embedding_function=embeddings,
    )
    pg_retriever = pgvector.as_retriever(
        search_type="mmr", search_kwargs={"k": 10, "fetch_k": 20}
    )

    embedding_memory = VectorStoreRetrieverMemory(
        retriever=pg_retriever,
        input_key="input",
        memory_key="embedding_key",
        exclude_input_keys=["history"],
    )

    chat_message_history = PostgresChatMessageHistory(
        session_id=str(channel),
        connection_string=str(os.getenv("CONNECTION_STRING")),
        table_name="chat_message_history",
    )

    chat_memory = ConversationSummaryBufferMemory(
        llm=llm,
        chat_memory=chat_message_history,
        input_key="input",
        max_token_limit=400,
    )
    chat_memory.prune()

    loaded[str(channel)] = CombinedMemory(memories=[chat_memory, embedding_memory])
    logging.info("Loaded memory for channel " + str(channel))


@bot.listen()
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # Make sure the message isn't a command that will be handled elsewhere
    if message.content.startswith("!"):
        return

    collection = str(message.guild.id if message.guild else message.channel.id)

    if collection not in loaded:
        loading_message = await message.channel.send("One moment, loading memory...")
        await load_all_memories([collection])
        await loading_message.delete()

    combined_memory = loaded[collection]

    # If in DMs or mentioned, respond
    if isinstance(message.channel, discord.DMChannel) or bot.user in message.mentions:
        async with message.channel.typing():
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, create_response, message, combined_memory
            )
            await message.channel.send(response)

    # React with an emoji
    await react_emoji_for_message(message, llm)


def create_response(message: discord.Message, memory: CombinedMemory) -> str:
    logging.info("Creating response for message " + str(message.content))
    conversation = ConversationChain(
        llm=llm_4, verbose=True, memory=memory, prompt=prompt
    )

    response = conversation.run(str(message.author) + ": " + message.content)

    logging.info("Response created: " + str(response))

    return response


# Run the bot
bot.run(str(os.getenv("DISCORD_TOKEN")))
