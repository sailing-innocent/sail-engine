from typing import Union 
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None

app = FastAPI()

@app.get("/")
def root():
    return {"hello": "fastapi hello!"}

@app.get("/say/{data}")
def say(data: str, q: Union[str, None] = None):
    return {"data": data, "q": q}

@app.post("/items/")
async def create_item(item: Item):
    print(item.name)
    return item