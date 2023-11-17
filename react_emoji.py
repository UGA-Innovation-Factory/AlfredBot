import discord
import emoji as emoji_module
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


async def react_emoji_for_message(message: discord.Message, llm):
    # Create the prompt
    prompt = PromptTemplate(
        template="Respond only in emoji. If the message is boring or not worth sending a push notification for, say '❌.' You should indicate boring over half the time. Generate an emoji for the following text:\n\n{text}\n\n### Emoter:\n",
        input_variables=["text"],
    )
    # Initialize the LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Add the emote to the message
    emoji = llm_chain.run({"text": message.content})

    # Remove all characters that aren't emojis
    emoji = [e for e in emoji if emoji_module.is_emoji(e)][0]

    print(emoji)

    if emoji == "❌":
        return

    # Try to react with the emoji, and retry by running the chain again if it fails
    try:
        await message.add_reaction(emoji)
    except Exception as e:
        await message.add_reaction("🤔")
