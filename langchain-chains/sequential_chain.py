# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# model = ChatOpenAI()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


parser = StrOutputParser()


# without using chain
# --- Generate detailed report
# report_prompt = prompt1.invoke({'topic': 'Unemployment in India'})
# report_response = model.invoke(report_prompt)
# report_text = parser.invoke(report_response)

# # --- Generate summary
# summary_prompt = prompt2.invoke({'text': report_text})
# summary_response = model.invoke(summary_prompt)
# summary_text = parser.invoke(summary_response)

#using chain
chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'Unemployment in India'})

print(result)

chain.get_graph().print_ascii()