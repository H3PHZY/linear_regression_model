from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from prediction import predict_wei, retrain_and_save


def _parse_cors_origins() -> list[str]:
    """
    Comma-separated env var e.g.:
      CORS_ORIGINS="https://my-render-app.onrender.com,http://localhost:3000"
    """
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080")
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(
    title="WEI Prediction API",
    description="Predict Women's Empowerment Index (WEI) and support model retraining.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)


class PredictRequest(BaseModel):
    women_empowerment_group: str = Field(
        ...,
        description="Women's Empowerment Group - 2022 (categorical; must match training categories).",
        min_length=1,
    )
    ggpi: float = Field(
        ...,
        description="Global Gender Parity Index (GGPI) - 2022",
        ge=0.0,
        le=1.0,
    )
    gender_parity_group: str = Field(
        ...,
        description="Gender Parity Group - 2022 (categorical; must match training categories).",
        min_length=1,
    )
    human_development_group: str = Field(
        ...,
        description="Human Development Group - 2021 (categorical; must match training categories).",
        min_length=1,
    )
    sdd_regions: str = Field(
        ...,
        description="Sustainable Development Goal regions (categorical; must match training categories).",
        min_length=1,
    )


class PredictResponse(BaseModel):
    predicted_wei: float = Field(..., description="Predicted Women's Empowerment Index (WEI) - 2022")


class RetrainResponse(BaseModel):
    best_model: str
    best_mse: float
    scores: Dict[str, Any]


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "WEI Prediction API. Swagger docs at /docs"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        predicted = predict_wei(
            women_empowerment_group=req.women_empowerment_group,
            ggpi=req.ggpi,
            gender_parity_group=req.gender_parity_group,
            human_development_group=req.human_development_group,
            sdd_regions=req.sdd_regions,
        )
        return PredictResponse(predicted_wei=predicted)
    except ValueError as e:
        # e.g. unseen categorical labels during LabelEncoder.transform
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed due to server error.")


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(file: UploadFile = File(...)) -> RetrainResponse:
    filename = (file.filename or "").lower()
    if not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    try:
        csv_bytes = await file.read()
        metrics = retrain_and_save(csv_bytes=csv_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Retraining failed due to server error.")

    return RetrainResponse(
        best_model=metrics["best_model"],
        best_mse=metrics["best_mse"],
        scores=metrics["scores"],
    )

