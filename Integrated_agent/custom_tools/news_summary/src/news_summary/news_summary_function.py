import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class NewsSummaryFunctionConfig(FunctionBaseConfig, name="news_summary"):
    """
    NAT function template. Please update the description.
    """
    per_page: int = 10
    api_key: str = ""
    discription: str = "Use this tool to get the latest news, and summarize the news in a concise manner."


@register_function(config_type=NewsSummaryFunctionConfig)
async def news_summary_function(
    config: NewsSummaryFunctionConfig, builder: Builder
):
    import os
    import requests
    import json


    async def _news_summary(question: str) -> str:
    # Search the web and get the requested amount of results
        url = "https://api.apitube.io/v1/news/everything"

        querystring = {"per_page":config.per_page, "api_key":config.api_key}
    
        response = requests.request("GET", url, params=querystring)
        news_results = json.loads(response.text)
        return news_results['results']

    yield FunctionInfo.from_fn(
    _news_summary,
    description=config.discription,
    )
