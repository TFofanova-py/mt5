import pandas as pd
from openai import AsyncClient
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed
import os
from enum import StrEnum
from ai_veto.binance import Binance
from typing import List


load_dotenv()

client = AsyncClient(api_key=os.getenv("TOGETHER_API_KEY"),
                     base_url="https://api.together.xyz/v1")

binance_client = Binance()

class Decision(StrEnum):
    accept = "Accept"
    reject = "Reject"


class AssistantAnswer(BaseModel):
    decision: Decision
    details: str



@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def llm_request(coin: str, direction: str, divergences: List[str], olhc_data: pd.DataFrame, verbose: bool = True) -> AssistantAnswer:
    olhc_data["timestamp"] = olhc_data.index
    olhc_data["timestamp"] = olhc_data["timestamp"].apply(lambda x: x.timestamp())
    messages = [{"role": "system", "content": "You are an expert in crypto trading."},
                {"role": "user", "content": await binance_client.get_prompt(coin=coin,
                                                                            direction=direction,
                                                                            divergences=divergences,
                                                                            ohlc_data=olhc_data.iloc[-10:].to_dict(orient="records"))}]
    response = await client.chat.completions.parse(
            model="openai/gpt-oss-20b",
            messages=messages,
            response_format=AssistantAnswer
        )

    result = AssistantAnswer.model_validate(response.choices[0].message.parsed)

    if verbose:
        print("AI verdict:", result.decision.value, "Details:", result.details)

    return result


