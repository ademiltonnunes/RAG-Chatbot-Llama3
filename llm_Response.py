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
#ChatBot
# # from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama

# libraries imported for translation
from deep_translator import GoogleTranslator, single_detection, batch_detection

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
    
    def __get_chroma_vectorstore(self):
        # Load the Chroma object from the file
        embedding = OllamaEmbeddings(model=model)
        vector_db = Chroma(persist_directory= persist_directory, embedding_function= embedding)

        return vector_db
        
    def __load_chatbot(self, k = 3, chain_type='stuff'):
        # Load the Chroma object from the file
        self.__db = self.__get_chroma_vectorstore()
       
       #Retrieve db        
        retriever = self.__db.as_retriever(search_type="similarity", search_kwargs={"k": k})

       # Create a ConversationBufferMemory
        memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True  # Return chat history as a list of messages
        )

        #Create QA
        qa = ConversationalRetrievalChain.from_llm(
            llm=Ollama(
                base_url=url,
                model=model
                ),
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=False,
            return_generated_question=False,
            memory=memory,
            output_key='answer'  # Specify the desired output key
        )
        return qa

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
                response = self.chatbot({'question': question, 'chat_history': []})

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
        return chatGptResponse
    
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
    # print(response.detect_language_chatGPT("嘿，怎么样"))
    # print(response.detect_language_chatGPT("Hēi, zěnme yàng?"))
    # print(response.detect_language_chatGPT("Oi, tudo bem?"))

if __name__ == "__main__":
    main()