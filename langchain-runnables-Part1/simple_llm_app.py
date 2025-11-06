from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Create a Prompt template
prompt = PromptTemplate(
    template='Suggest a catchy blog title about {topic}',
    input_variables=['topic']
)

# Define the input 
topic = input('Enter a topic')

#Format the prompt manually using PromptTemplate
formatted_prompt = prompt.format(topic=topic)

#Call the LLM directly
blog_title = llm.predict(formatted_prompt)

#Print the output 
print("Generating Blog Title:", blog_title)