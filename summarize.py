import validators
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains import LLMChain
from bs4 import BeautifulSoup


async def summarize(ctx, url: str, llm):
    # Validate URL
    if not validators.url(url):  # type: ignore
        await ctx.send("Please enter a valid URL.")
        return

    # Create headers for the URL request
    headers = {"User-Agent": "Mozilla/5.0"}

    # Initialize the loader with the URL and headers
    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False, headers=headers)

    try:
        # Load data from the URL
        data = loader.load()[0]
    except Exception as e:
        await ctx.send(f"An error occurred while loading the URL: {e}")
        return

    # Extract text content from the HTML
    soup = BeautifulSoup(data.page_content, "html.parser")
    text_content = soup.get_text()

    # Create the summarization prompt
    prompt = PromptTemplate(
        template="Write a summary of the following in 250-300 words:\n\n{text}\n\nSummary:\n",
        input_variables=["text"],
    )

    # Initialize the summarization process
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Create the data dictionary for the LLM chain
    data_for_chain = {"text": text_content}

    try:
        # Run the LLM chain to get the summary
        async with ctx.typing():
            summary = llm_chain.run(data_for_chain)
    except Exception as e:
        await ctx.send(f"An error occurred while summarizing the URL: {e}")
        return

    # Send the summary to the Discord context
    await ctx.send(summary)

    # React with a thumbs up
    await ctx.message.add_reaction("👍")
