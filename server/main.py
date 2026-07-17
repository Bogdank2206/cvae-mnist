import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.services.inference import load_model

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="CVAE Generator", version="0.1.0", lifespan=lifespan)

app.include_router(router)
