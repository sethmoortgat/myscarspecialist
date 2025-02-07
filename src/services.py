from src.prompts import (
	template_text_system,
	context_template_text,
	question_to_prompt_system_text,
	question_to_prompt_user_text,
)

def get_context(
	query,
    vectorstore,
    n_chunks=3,
    filters = None,
):
	chunks = vectorstore.max_marginal_relevance_search(
		query,
		k=n_chunks,
		filter=filters,
	)

	context = ""
	for _chunk in chunks:
		summary = "###\n"+ _chunk.page_content + "\n This info was retrieved from: " + _chunk.metadata["url"] + "\n###\n"
		context+=summary
	
	return context


def get_llm_response(
	messages,
	llm,
):
	response = llm.invoke(messages)
	return response.content


def transfor_user_question(last_question,messages,llm):
	messages_flat = ""
	for message in messages:
		role = message['role']
		content = message['content']
		messages_flat = messages_flat + f"{role}: {content} \n\n"
	
	system_prompt = question_to_prompt_system_text
	user_prompt = question_to_prompt_user_text.format(chat_history = messages_flat,
														question = last_question)
														
	messages_total = [
		{"role":"system", "content":system_prompt},
		{"role":"user", "content":user_prompt},
	]
	
	resp = get_llm_response(
		messages_total,
		llm,
	)
	return resp