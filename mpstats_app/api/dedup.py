from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from mpstats_app.api.dependencies import get_dedup_service
from mpstats_app.schemas import DedupProductSplitPayload, DedupRunPayload, DedupSettingsPayload
from pipeline.services.dedup import DedupService


router = APIRouter(prefix="/api/dedup", tags=["dedup"])


def _handle(call):
    try:
        return call()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/settings")
def get_dedup_settings(service: DedupService = Depends(get_dedup_service)) -> dict[str, object]:
    return service.get_settings()


@router.put("/settings")
def put_dedup_settings(
    payload: DedupSettingsPayload,
    service: DedupService = Depends(get_dedup_service),
) -> dict[str, object]:
    return _handle(lambda: service.save_settings(payload.model_dump()))


@router.get("/eligible-categories")
def eligible_categories(
    project_name: str,
    service: DedupService = Depends(get_dedup_service),
) -> dict[str, object]:
    return _handle(lambda: service.eligible_categories(project_name=project_name))


@router.get("/runs")
def list_runs(
    project_name: str | None = None,
    category_key: str | None = None,
    service: DedupService = Depends(get_dedup_service),
) -> dict[str, object]:
    return _handle(lambda: service.list_runs(project_name=project_name, category_key=category_key))


@router.get("/products")
def products_browser(
    project_name: str,
    category_key: str | None = None,
    level: str = "expanded",
    query: str | None = None,
    limit: int = 500,
    service: DedupService = Depends(get_dedup_service),
) -> dict[str, object]:
    return _handle(
        lambda: service.products_browser(
            project_name=project_name,
            category_key=category_key,
            level=level,
            query_text=query,
            limit=limit,
        )
    )


@router.get("/products/export")
def export_products(
    project_name: str,
    category_key: str | None = None,
    level: str = "expanded",
    service: DedupService = Depends(get_dedup_service),
) -> FileResponse:
    target = _handle(lambda: service.export_products(project_name=project_name, category_key=category_key, level=level))
    return FileResponse(target, filename=target.name)


@router.post("/products/split")
def split_product_from_family(
    payload: DedupProductSplitPayload,
    service: DedupService = Depends(get_dedup_service),
) -> dict[str, object]:
    return _handle(lambda: service.split_product_from_family(run_id=payload.run_id, node_id=payload.node_id, note=payload.note))


@router.post("/runs")
def start_runs(
    payload: DedupRunPayload,
    service: DedupService = Depends(get_dedup_service),
) -> dict[str, object]:
    return _handle(
        lambda: service.start_runs(
            project_name=payload.project_name,
            category_keys=payload.category_keys,
            wait=payload.wait,
        )
    )


@router.get("/runs/{run_id}")
def get_run(run_id: str, service: DedupService = Depends(get_dedup_service)) -> dict[str, object]:
    return _handle(lambda: service.get_run(run_id))


@router.get("/runs/{run_id}/export")
def export_artifact(
    run_id: str,
    artifact: str = "groups",
    service: DedupService = Depends(get_dedup_service),
) -> dict[str, object]:
    return _handle(lambda: service.export_artifact(run_id=run_id, artifact=artifact))
