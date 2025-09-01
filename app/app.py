import os
import re
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --------------- ENVIRONMENT CONFIGURATION ---------------
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
 load_dotenv(dotenv_path=env_path)
else:
    print(f"Warning: .env file not found at {env_path}. Environment variables may not be set correctly.")
    
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

import openai
client = openai.OpenAI( )

# --------------- FASTAPI & Pydantic Model -----------------
app = FastAPI(
    title="ABAP Declaration Parser LLM",
    description="API: send ABAP source code, get LLM-extracted declaration table as JSON.",
    version="2.1.0"
)

# Updated request model with extra fields
class ABAPCode(BaseModel):
    code: str
    pgm_name: str
    inc_name: str
    type: str

# --------------- PROMPT GENERATION ------------------------
def abap_lm_prompt(code: str) -> str:
    return (
        "You are an ABAP code analysis assistant.\n"
        "Analyze the following ABAP code and extract only SELECT-OPTIONS and PARAMETERS declarations (selection screen parameters).\n"
        "For each, return an object with keys: type, name, object, description "
        '(in which "type" is the declaration keyword, "name" is the identifier, "object" is the technical reference, and "description" is a one-sentence explanation).\n'
        "**Return the result as a raw JSON array, with NO prose, no markdown, no code block, and no explanation. Only output a single array.**\n\n"
        f'{code}'
    )

# --------------- JSON EXTRACTION --------------------------
def extract_json_from_text(text):
    try:
        # 1. Common: Wrapped in markdown ```json ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if match:
            snippet = match.group(1)
            return json.loads(snippet)
        # 2. Fallback: First array anywhere
        match = re.search(r"(\[\s*{[\s\S]+?}\s*])", text)
        if match:
            snippet = match.group(1)
            return json.loads(snippet)
        # 3. As last resort, try to load entire response
        return json.loads(text)
    except Exception as e:
        print("!! Failed to extract JSON:", e)
        print("!! Offending content:")
        print(text)
        return []

# --------------- LLM EXTRACTION FUNCTION ------------------
import asyncio

async def abap_llm_declarations(code: str):
    prompt = abap_lm_prompt(code)
    from starlette.concurrency import run_in_threadpool

    def _call_openai():
        return client.chat.completions.create(
            model="gpt-4.1",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful code assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0
        )
    response = await run_in_threadpool(_call_openai)
    resp_text = response.choices[0].message.content
    print("===== LLM raw output begin =====")
    print(resp_text)
    print("===== LLM raw output end   =====")

    declarations = extract_json_from_text(resp_text)
    return declarations

# --------------- API ENDPOINT/CONTROLLER ------------------
@app.post("/abap/selectionscreen")
async def abap_declarations_api(input: ABAPCode, request: Request):
    pgm_name = input.pgm_name
    inc_name = input.inc_name
    type_field = input.type
    code = input.code
    

    if not isinstance(code, str) or not code.strip():
        raise HTTPException(status_code=400, detail="Missing or empty 'code' field.")

    declarations = await abap_llm_declarations(code)

    # Remove 'type' from each declaration returned by the LLM
    for decl in declarations:
        decl.pop("type", None)

    # Response body with all four fields
    response_body = {
        "pgm_name": pgm_name,
        "inc_name": inc_name,
        "type": type_field,
        "selectionscreen": declarations
    }

    # Return in body only, matching your sample format
    return JSONResponse(content=response_body)
