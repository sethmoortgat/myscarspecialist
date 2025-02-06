from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

import streamlit as st
import openai
import os
import hmac



lang_dict = {
	"🇧🇪  Nederlands":"NL",
	"🇬🇧  English":"EN"
}

template_text_system = """ You are a friendly assistant that helps people who are browsing a website with information on scar treatments.
You are polite, provide extensive accurate answers, and point the user to the right location for more information.
Please make sure your answer is provided in {language}.

You have to answer a question that you find below, but only using information in the context below.
Do not use any other information and make sure your answer is almost an exact copy of the relevant text in the context.
The provided context is split in different chunks of information delimited by triple '#', and at the end of each
piece of context you find a urls where the info is retrieved from. You are allowed to combine information from
different parts of the context into one consistent and complete answer.

If the question is completely unrelated to the treatment of scars, do NOT make up an answer but instead reply with:
'Sorry, this information can not be found on the website.'. If however you can not find an exact answer in the context, 
but you find some related information, you can still give a reply acknowleding that it might not exactly answer their question,
but more info might be available on the website.

If you give an answer, end your answer by stating on which website this info can be found, which is given at the end of each piece of context.
Make sure to give the entire link, starting with 'https:'
Add the URL in the following form: "You can read more about <topic_the_question_was_about> on: https://..."
You can use the context of the entire chat history to answer any follow-up questions
"""

context_template_text = "The following context has been added to the conversation: {context}"

question_to_prompt_system_text = """Your task is to, given a chat history and the latest user question, which might reference context in the chat history, 
to formulate a standalone question which can be understood without the chat history.
This question will be used to retrieve relevant context to answer the latest user question.
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

Only return the reformulated question, do not say anything else.
Return only a single consistent answer that is precise to the request of the user.
"""

question_to_prompt_user_text = """
Chat history:
{chat_history}

Latest user question:
{question}
"""

	



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

def main():

	def check_password():
		"""Returns `True` if the user had the correct password."""

		def password_entered():
			"""Checks whether a password entered by the user is correct."""
			if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
				st.session_state["password_correct"] = True
				del st.session_state["password"]  # Don't store the password.
			else:
				st.session_state["password_correct"] = False

		# Return True if the passward is validated.
		if st.session_state.get("password_correct", False):
			return True

		# Show input for password.
		st.text_input(
			"Password", type="password", on_change=password_entered, key="password"
		)
		if "password_correct" in st.session_state:
			st.error("😕 Password incorrect")
		return False


	if not check_password():
		st.stop()  # Do not continue if check_password is not True.
	

	
	def new_question():
		st.session_state.new_question = True
		st.session_state.question = ''
		st.session_state.context = ''
		st.session_state.messages = []
		st.session_state.chat_history = []
		st.session_state.n_questions = 0
	
	def existing_question():
		st.session_state.new_question = False
	
	def add_user_input():
		st.session_state.messages.append({'role':'user', 'content':st.session_state.chat_input})
		st.session_state.chat_history.append({'role':'user', 'content':st.session_state.chat_input})
		
	def change_language():
		st.session_state.language = lang_dict[st.session_state.new_language]
		new_question()
	
	st.set_page_config(
		layout="centered",
		page_icon="./SCARBOT_AVATAR.png",
	) # centered wide
	
	if "language" not in st.session_state.keys():
		st.session_state.language = 'NL'
	
	if "question" not in st.session_state.keys():
		st.session_state.question = ''
	
	if "context" not in st.session_state.keys():
		st.session_state.context = ''
	
	if "new_question" not in st.session_state.keys():
		st.session_state.new_question = True
	
	if "embedding_function" not in st.session_state.keys():
		st.session_state.embedding_function = OpenAIEmbeddings(
			openai_api_key=st.secrets["openai_api_key"],
			model="text-embedding-3-large",
		)
	
	@st.cache_resource
	def get_qdrant_client():
		client = QdrantClient(path="./myscarspecialist_qdrant_db")
		return client
	
	if "vectorstore" not in st.session_state.keys():
		st.session_state.vectorstore = QdrantVectorStore(
			client=get_qdrant_client(),
			collection_name="myscarspecialist",
			embedding=st.session_state.embedding_function,
		)
		
	if "openai_client" not in st.session_state.keys():
		st.session_state.openai_client = ChatOpenAI(
			model="gpt-4o",
			temperature=0.2,
			max_tokens=None,
			timeout=None,
			max_retries=2,
			openai_api_key=st.secrets["openai_api_key"]
		)

	if "messages" not in st.session_state.keys():
		st.session_state.messages = [] 
	
	if "chat_history" not in st.session_state.keys():
		st.session_state.chat_history = [] 
	
	
	
	
	
	header = st.container()
	with header:
		col1, col2 = st.columns([5,5])
		with col1:
			st.button("New question" if st.session_state.language == 'EN' else "Nieuwe vraag" , on_click=new_question, type="primary")
		with col2:
			st.selectbox(
				label="language",
				options=(lang_dict.keys()),
				key="new_language",
				on_change=change_language,
				label_visibility="collapsed",
			)
	header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

	### Custom CSS for the sticky header top: 3.755rem;
	st.markdown(
		"""
	<style>
		div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
			position: sticky;
			top: 3.755rem;
			background-color: white;
			z-index: 999;
		}
		.fixed-header {
			border-bottom: 1px solid black;
		}
	</style>
		""",
		unsafe_allow_html=True
	)
	
	

	if st.session_state.new_question:	
		col1, col2, col3, col4 = st.columns([1,3,10,1])
		with col2:
			st.image("./SCARBOT_AVATAR.png",width=200)
		with col3: 
			st.markdown('## Hi, my name is Scarbot!' if st.session_state.language == 'EN' else "## Hallo, mijn naam is Scarbot!")
			st.markdown('## How can I help you?' if st.session_state.language == 'EN' else "## Hoe kan ik je helpen?")

		label = 'Ask your question' if st.session_state.language == 'EN' else "Stel je vraag"

		st.text_input("original question", placeholder = label, key='question', on_change=existing_question, label_visibility="collapsed")

		
	else:
		
		if len(st.session_state.messages) == 0:
			# First initialise the system prompt
			system_prompt = template_text_system.format(language = "dutch" if st.session_state.language == "NL" else "english")
			st.session_state.messages.append({'role':'system', 'content':system_prompt})
			st.session_state.chat_history.append({'role':'system', 'content':system_prompt})
			
			# Then add the context prompt
			with st.spinner('Browsing website...' if st.session_state.language == 'EN' else "Website doorzoeken..."):
				st.session_state.context = get_context(
					st.session_state.question,
					st.session_state.vectorstore,
					n_chunks=3,
					filters=models.Filter(
						must=[
							models.FieldCondition(
								key="metadata.language",
								match=models.MatchValue(value='nl' if st.session_state.language == "NL" else 'en'),
							),
						]
					),
				)
			context_prompt = context_template_text.format(context = st.session_state.context)
			st.session_state.messages.append({'role':'system', 'content':context_prompt})
			st.session_state.chat_history.append({'role':'system', 'content':context_prompt})
			
			# Finally add the user question
			st.session_state.messages.append({'role':'user', 'content':st.session_state.question})
			st.session_state.chat_history.append({'role':'user', 'content':st.session_state.question})
		
		else:
			last_question = st.session_state.messages[-1]["content"]
			transformed_last_question = transfor_user_question(last_question,st.session_state.messages,st.session_state.openai_client)

			with st.spinner('Browsing website...' if st.session_state.language == 'EN' else "Website doorzoeken..."):
				follow_up_context = get_context(
					transformed_last_question,
					st.session_state.vectorstore,
					n_chunks=2,
					filters=models.Filter(
						must=[
							models.FieldCondition(
								key="metadata.language",
								match=models.MatchValue(value='nl' if st.session_state.language == "NL" else 'en'),
							),
						]
					),
				)
			context_prompt = context_template_text.format(context = follow_up_context)
			st.session_state.messages.append({'role':'system', 'content':context_prompt})
			st.session_state.chat_history.append({'role':'system', 'content':context_prompt})

		with st.spinner('Composing answer...' if st.session_state.language == 'EN' else "Antwoord genereren..."):	
			answer = get_llm_response(
				st.session_state.messages,
				st.session_state.openai_client,
			)
		st.session_state.messages.append({'role':'assistant', 'content':answer})
		st.session_state.chat_history.append({'role':'assistant', 'content':answer})

		
			
		# show the chat history on screen
		with st.container():
			for idx, message in enumerate(st.session_state.chat_history):
				if message["role"]=="system": continue
				else: 
					if message["role"]=="assistant":icon="./avatar_icon.png"
					else: icon="./user_icon.png"
					with st.chat_message(message["role"],avatar=icon): st.write(message["content"])
		

		st.chat_input("Enter your follow-up question..." if st.session_state.language == 'EN' else "Zet het gesprek verder...", 
			key="chat_input", on_submit = add_user_input)

		
	
	
	
	

if __name__ == '__main__':
	main()
