import uvicorn
from fastapi import FastAPI, Form
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from parse_text import parse_text_api

app = FastAPI()


@app.post("/parse_text")
async def parse(text: str = Form(...)):
    parsed_text = jsonable_encoder(parse_text_api(text))
    response = JSONResponse(content=parsed_text)
    return response

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True, access_log=False)
