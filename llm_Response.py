import os
import sys
import re
import ollama
sys.path.append('../..')

# # __import__('pysqlite3')
# # sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#Embedding
from langchain_community.embeddings import OllamaEmbeddings
#Vectorstores
from langchain_community.vectorstores import Chroma
# BufferMemory
from langchain.memory import ConversationBufferMemory
# Template
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
# LLM
from langchain_community.llms import Ollama
#ChatBot Retriever
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA



# libraries imported for translation
from deep_translator import GoogleTranslator, single_detection

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
model  = os.environ['MODEL']
url  = os.environ['URL']
language_detection_model = os.environ['LANGUAGE_DETECTOR_API_KEY']
persist_directory = 'docs/chroma/'

class LLMResponse:

    def __init__(self) -> None:
        self.language = None
        self.__db = None
        self.chatbot = None
        self.chat_history = []
    
    def __get_chroma_vectorstore(self):
        # Load the Chroma object from the file
        embedding = OllamaEmbeddings(model=model)
        vector_db = Chroma(persist_directory= persist_directory, embedding_function= embedding)

        return vector_db
        
    def __load_chatbot(self, k = 3 , chain_type='stuff'):
        # Load the Chroma object from the file
        self.__db = self.__get_chroma_vectorstore()
       
       #Retrieve db        
        retriever = self.__db.as_retriever(search_type="similarity", search_kwargs={"k": k})

       # Create a ConversationBufferMemory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,  # Return chat history as a list of messages
            input_key="question"
        )

        # Define the system message template
        system_template = """You are a knowledgeable chatbot for Customer Support for SFBU, which stands for San Francisco Bay University.
        You are here to help with questions of the user. 
        Your tone should be professional and informative, but not too wordy. For example, if you greated the user, don't great them again in the answer. Make responses shortest as possible.
        If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
        Don't guess anything about the user, just answer their question.
        ----------------
        {context}"""

        # Create the chat prompt templates
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
        qa_prompt = ChatPromptTemplate.from_messages(messages)

        # LLM
        llm = Ollama(
            base_url=url,
            model=model
        )
 
        #Create QA
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=False,
            return_generated_question=False,
            memory=memory,
            output_key='answer',
            verbose=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )

        return qa

    def __load_chatbot2(self, k = 6, chain_type='stuff'):
        # Load the Chroma object from the file
        self.__db = self.__get_chroma_vectorstore()
        
        #Retrieve db        
        retriever = self.__db.as_retriever(search_type="similarity", search_kwargs={"k": k})



        # Create Template
        prompt_template = """You are a knowledgeable chatbot for Customer Support for SFBU, which stands for San Francisco Bay University.
        You are here to help with questions of the user. 
        Your tone should be professional and informative, but not too wordy. For example, if you greated the user, don't great them again in the answer. Make responses shortest as possible.
        If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
        Don't guess anything about the user, just answer their question.
        
            Context: {context}
            History: {history}

            User: {question}
            Chatbot:"""
        
        template = PromptTemplate(
            input_variables=["history", "context", "question"],
            template= prompt_template
        )

        # Create a ConversationBufferMemory
        memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,  # Return chat history as a list of messages
            input_key="question"
        )

        self.chat_history = []

        # LLM
        llm = Ollama(
            base_url=url,
            model=model
        )

        #Create QA
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": template,
                "memory": memory
            }
        )

        return qa
    
    def chat2(self, question, includeAudio):
        try:

            #Load chatbot
            if self.chatbot == None:
                self.chatbot = self.__load_chatbot() 

            #saving original question
            original_question = question

            #detecting language if no audio question
            if not includeAudio:
                self.language = self.__detect_language_chatGPT(question)

                #translate language to english if language isn't in English
                if self.language != 'en':                    
                    question = self.__translator(question, source_language= self.language, target_language='en')          

            # Check for prompt injection and moderation before answing question
            if self.__detect_prompt_injection(question):
                answer = "Your request has been flagged as potential prompt injection and cannot be processed."
            else:
                #Submit question
                user_message = {"role": "user", "message": question}
                self.chat_history.append(user_message)
                
                response = self.chatbot({"query": question, "history": self.chat_history})
                answer = response.get('result')
                
                chatbot_message = {"role": "assistant", "message": response['result']}
                self.chat_history.append(chatbot_message)

            #Translate response if not audio
            if not includeAudio:
                if self.language != 'en':                    
                    answer = self.__translator(answer, source_language= 'en', target_language= self.language)

            #Retrieve Answer
            return {"question": original_question, "answer": answer}
        except Exception as e:
            return {"answer": f"An error occurred: {str(e)}"}      

    def chat(self, question, includeAudio):
        try:

            #Load chatbot
            if self.chatbot == None:
                self.chatbot = self.__load_chatbot() 

            #saving original question
            original_question = question

            #detecting language if no audio question
            if not includeAudio:
                self.language = self.__detect_language_chatGPT(question)

                #translate language to english if language isn't in English
                if self.language != 'en':                    
                    question = self.__translator(question, source_language= self.language, target_language='en')          

            # Check for prompt injection and moderation before answing question
            if self.__detect_prompt_injection(question):
                response =  {"answer": "Your request has been flagged as potential prompt injection and cannot be processed."}
            else:
                #Submit question
                response = self.chatbot({'question': question})

            #Translate response if not audio
            if not includeAudio:
                if self.language != 'en':                    
                    answer = self.__translator(response.get('answer'), source_language= 'en', target_language= self.language)
                    response =  {"answer": answer}

            #Retrieve Answer
            answer = response.get('answer')
            return {"question": original_question, "answer": answer}
        
        except Exception as e:
            return {"answer": f"An error occurred: {str(e)}"}      

    def __generate_answer_llm(self, user_prompts:list[str], system_prompt:str =""):
        messages = []

        # Add user messages
        for prompt in user_prompts:
            messages.append({'role': 'user', 'content': prompt})

        if system_prompt !="":
            messages.append({"role": "system", "content":system_prompt})


        response = ollama.chat(model=model, messages=messages)
        return response['message']['content']

    ########################## PROMPT INJECTION ###########   
    def __detect_prompt_injection(self, question):
        try:
            # Perform prompt injection check
            return self.__prompt_injection_check(question)
        except Exception as e:
            # Handle translation errors
            print(f"Prompt Injection: {str(e)}")
            return True  # Consider it flagged in case of translation errors

    def __prompt_injection_check(self, question):
        # Check for specific patterns indicative of prompt injection in English
        prompt_injection_patterns = [
            r"\bignore\b",
            r"\bdisregard\b.*\binstructions\b",
            r"\boverride\b.*\binstructions\b",
            r"\bmalicious\b.*\binstructions\b",
            r"\bconflicting\b.*\binstructions\b",
            # Add more English patterns as needed
        ]

        for pattern in prompt_injection_patterns:
            if re.search(pattern, question, flags=re.IGNORECASE):
                return True  # Flagged for prompt injection

        return False  # Not flagged for prompt injection

    #######################Language detection###########    
    def __detect_language_chatGPT(self, question:str):
        system_prompt = """
        You are an language detector assistant, you detects language of the user's text. \
        The user will send you a phrase or a single word. \
        You respond with two letters only. \
        For example, 'en' if the prompt is in English, \
        or 'pt' if it is in Portuguese, \
        or 'fr' when it is in French,\
        or two letters that represent any other language. \
        However, if the prompt is in chinese (simplified) you have to return 'zh-CN' \
        or if chinese (traditional) you return 'zh-TW'. \
        Output has to be two letters only with no punctuation, except 'zh-CN' for chinese (simplified)\
        or 'zh-TW' chinese (traditional) with no punctuation.
        Case you don't recognize all prompt words, consider other words. \
        Case you don't recognize the language, return 'en'.
        Don't write nothing else, only the language of the user's text, please.
        """
        q_a_pair = f"Detect the language of the following prompt: {question} "
        prompts = []
        prompts.append(q_a_pair)

        chatGptResponse=self.__generate_answer_llm(prompts, system_prompt)

        # Check if the response is a valid language acronym
        if not self.__is_valid_language_acronym(chatGptResponse):
            chatGptResponse = 'en'
        
        return chatGptResponse
    
    def __is_valid_language_acronym(self, acronym):
        pattern = r'^(af|sq|am|ar|hy|as|ay|az|bm|eu|be|bn|bho|bs|bg|ca|ceb|ny|zh-CN|zh-TW|co|hr|cs|da|dv|doi|nl|en|eo|et|ee|tl|fi|fr|fy|gl|ka|de|el|gn|gu|ht|ha|haw|iw|hi|hmn|hu|is|ig|ilo|id|ga|it|ja|jw|kn|kk|km|rw|gom|ko|kri|ku|ckb|ky|lo|la|lv|ln|lt|lg|lb|mk|mai|mg|ms|ml|mt|mi|mr|mni-Mtei|lus|mn|my|ne|no|or|om|ps|fa|pl|pt|pa|qu|ro|ru|sm|sa|gd|nso|sr|st|sn|sd|si|sk|sl|so|es|su|sw|sv|tg|ta|tt|te|th|ti|ts|tr|tk|ak|uk|ur|ug|uz|vi|cy|xh|yi|yo|zu)$'
        return bool(re.match(pattern, acronym))
    
    def detect_language_chatGPT(self, text:str):
        detected_language = single_detection(
            text = text,
            api_key = language_detection_model
            
            )
        
        if detected_language == 'und':
            detected_language = 'en'
        
        return detected_language

    ########################## Translating the Answer ###########
    def __translator(self, text: str, source_language: str, target_language: str):
        translated_text = GoogleTranslator(source=source_language, target=target_language).translate(text)

        return translated_text

def main():
    response = LLMResponse()
    # print(response.is_valid_language_acronym("en"))
    # print(response.detect_language_chatGPT("嘿，怎么样"))
    # print(response.detect_language_chatGPT("Hēi, zěnme yàng?"))
    # print(response.detect_language_chatGPT("Oi, tudo bem?"))

if __name__ == "__main__":
    main()