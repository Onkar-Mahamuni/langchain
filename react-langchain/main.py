from typing import Union, List

from dotenv import load_dotenv
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool, tool
from langchain.tools.render import render_text_description
from langchain_ollama import ChatOllama

load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """
    Returns the length of a text by characters.
    """
    return len(text)

if __name__ == "__main__":
    print("Hello, ReAct LangChain!")
    
    tools = [get_text_length]

    template = """
      Answer the following questions as best you can. You have access to the following tools:

      {tools}

      Use the following format:

      Question: the input question you must answer
      Thought: you should always think about what to do
      Action: the action to take, should be one of [{tool_names}]
      Action Input: the input to the action
      Observation: the result of the action
      ... (this Thought/Action/Action Input/Observation can repeat N times)
      Thought: I now know the final answer
      Final Answer: the final answer to the original input question

      Begin!

      Question: {input}
      Thought:
    """

    prompt = PromptTemplate.from_template(template=template).partial(tools=render_text_description(tools), tool_names=", ".join([tool.name for tool in tools]))

    # llm = ChatOpenAI(temperature=0, stop= ["\nObservation", "Observation"], model="gpt-4o-mini")
    llm = ChatOllama(model="llama3.1:8b", stop= ["\nObservation", "Observation"], temperature=0)

    agent = {"input": lambda x:x["input"]} | prompt | llm | ReActSingleInputOutputParser()

    res = agent.invoke({"input": "What is the length of 'Dog' in characters?"})

    print(res)