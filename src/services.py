from src.prompts import (
    question_to_prompt_system_text,
    question_to_prompt_user_text,
)
import logging


def get_context(
    query,
    qdrant_client,
    collection_name,
    embedding_function,
    n_chunks=3,
    filters=None,
):
    logging.info(f"Using filter: {filters}")

    query_vector = embedding_function.embed_query(query)

    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=n_chunks,
    )

    logging.info(f"Found {len(results.points)} chunks for query: {query}")

    if results.points:
        source_urls = [point.payload.get("url", "unknown") for point in results.points]
        logging.info(f"Chunks retrieved from URLs: {source_urls}")

    context = ""
    for point in results.points:
        text = point.payload.get("text", "")
        url = point.payload.get("url", "unknown source")
        summary = "###\n" + text + "\n This info was retrieved from: " + url + "\n###\n"
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
