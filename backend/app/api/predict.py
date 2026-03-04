from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.ml.inference import predict_point, predict_risk_map

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.get("/point")
def predict_point_endpoint(lat: float, lon: float):
    try:
        return predict_point(lat=lat, lon=lon)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/risk-map")
def predict_risk_map_endpoint():
    try:
        risk_map_path = predict_risk_map()
        return FileResponse(str(risk_map_path), media_type="image/tiff", filename=risk_map_path.name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
