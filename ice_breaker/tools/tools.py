from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_tavily import TavilySearch


def get_profile_url_tavily(name: str):
    """Searches for Linkedin or Twitter Profile Page."""
    search = TavilySearchResults()
    res = search.run(f"{name}")
    return res[0]["url"]
    # return TavilySearch(max_results=5, topic="general")