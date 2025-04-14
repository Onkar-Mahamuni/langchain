from dotenv import load_dotenv
import os
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from tools.tools import get_profile_url_tavily

def lookup(name: str) -> str:

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    template = """
      Given the name of a person: {name_of_person}, I want you to find a link to their Twitter profile and extract from it their username.
      In your final answer, include only the person's username
      """
    
    prompt_template = PromptTemplate(input_variables=["name_of_person"], template=template)
    
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 twitter profile page",
            func=get_profile_url_tavily,
            description="Useful for when you need to get the twitter page url"
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools_for_agent, react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(input={"input": prompt_template.format_prompt(name_of_person=name)})

    twitter_username = result["output"]
    return twitter_username

if __name__ == "__main__":
    print(lookup(name="Eden Marco"))