from src.prompts import (
    question_to_prompt_system_text,
    question_to_prompt_user_text,
)
import logging


def get_context(
    query,
    vectorstore,
    n_chunks=3,
    filters=None,
):
    # Debug log what filter is being used
    logging.info(f"Using filter: {filters}")

    # Try to retrieve chunks
    chunks = vectorstore.max_marginal_relevance_search(
        query,
        k=n_chunks,
        filter=filters,
    )

    # Log the number of chunks found
    logging.info(f"Found {len(chunks)} chunks for query: {query}")

    # If no chunks found, try without filter
    if len(chunks) == 0 and filters is not None:
        logging.warning(f"No chunks found with filter. Trying without filter...")
        chunks = vectorstore.max_marginal_relevance_search(
            query,
            k=n_chunks,
            filter=None,
        )
        logging.info(f"Found {len(chunks)} chunks without filter")

    # Collect and log the source URLs
    if chunks:
        source_urls = [chunk.metadata["url"] for chunk in chunks]
        logging.info(f"Chunks retrieved from URLs: {source_urls}")

        # Debug the metadata structure of the first chunk
        if len(chunks) > 0:
            logging.info(f"First chunk metadata: {chunks[0].metadata}")

    context = ""
    for _chunk in chunks:
        summary = (
            "###\n"
            + _chunk.page_content
            + "\n This info was retrieved from: "
            + _chunk.metadata["url"]
            + "\n###\n"
        )
        context += summary

    return context


def get_llm_response(
    messages,
    llm,
):
    response = llm.invoke(messages)
    return response.content


def transfor_user_question(last_question, messages, llm):
    messages_flat = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        messages_flat = messages_flat + f"{role}: {content} \n\n"

    system_prompt = question_to_prompt_system_text
    user_prompt = question_to_prompt_user_text.format(
        chat_history=messages_flat, question=last_question
    )

    messages_total = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    resp = get_llm_response(
        messages_total,
        llm,
    )
    return resp
