# parser.py
import json
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

class StructuredResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float

def parse_response(text: str) -> StructuredResponse:
    clean = text.strip()
    parsed = json.loads(clean)
   
    
    if parsed.get("confidence", 0) == 0 and parsed.get("answer", "") != "I cannot find this in the provided context":
        parsed["confidence"] = 0.8
    return StructuredResponse(**parsed)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def get_structured_response(client, messages: list) -> StructuredResponse:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    text = response.choices[0].message.content
    return parse_response(text)
