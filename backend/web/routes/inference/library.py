from __future__ import annotations

import json
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .schemas import SaveHardwareRequest
from backend.utils.log import logger

router = APIRouter(prefix="/api/library", tags=["library"])

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "inference", "data")


# ── Operators ──────────────────────────────────────────

@router.get("/operators")
async def list_operators():
    """列出所有内置算子名称。"""
    ops_dir = os.path.join(_DATA_DIR, "operators")
    if not os.path.isdir(ops_dir):
        return []
    return sorted(
        f.replace(".json", "")
        for f in os.listdir(ops_dir)
        if f.endswith(".json")
    )


@router.get("/operators/{name}")
async def get_operator(name: str):
    """获取指定算子的 JSON 定义。"""
    path = os.path.join(_DATA_DIR, "operators", f"{name}.json")
    if not os.path.exists(path):
        raise HTTPException(404, f"算子 '{name}' 不存在")
    with open(path) as f:
        return json.load(f)


# ── Models ─────────────────────────────────────────────

class SaveModelRequest(BaseModel):
    name: str
    architecture: str = "custom"
    num_layers: int = 0
    hidden_dim: int = 4096
    vocab_size: int = 128256
    layers: list[dict] = []


@router.get("/models")
async def list_models():
    """列出所有已保存的模型名称。"""
    models_dir = os.path.join(_DATA_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    return sorted(
        f.replace(".json", "")
        for f in os.listdir(models_dir)
        if f.endswith(".json")
    )


@router.get("/models/{name}")
async def get_model(name: str):
    """获取指定模型的 JSON 定义。"""
    path = os.path.join(_DATA_DIR, "models", f"{name}.json")
    if not os.path.exists(path):
        raise HTTPException(404, f"模型 '{name}' 不存在")
    with open(path) as f:
        return json.load(f)


@router.post("/models")
async def save_model(req: SaveModelRequest):
    """保存模型 JSON 到本地文件。"""
    models_dir = os.path.join(_DATA_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{req.name}.json")
    with open(path, "w") as f:
        json.dump(req.model_dump(), f, indent=2, ensure_ascii=False)
    logger.info("saved model '%s' (%d layers)", req.name, len(req.layers))
    return {"status": "ok", "name": req.name}


@router.delete("/models/{name}")
async def delete_model(name: str):
    """删除指定模型。"""
    path = os.path.join(_DATA_DIR, "models", f"{name}.json")
    if os.path.exists(path):
        os.remove(path)
        logger.info("deleted model '%s'", name)
    else:
        logger.warning("delete model '%s': file not found", name)
    return {"status": "ok"}


# ── Hardware ───────────────────────────────────────────

def _find_hardware_file(hw_name: str) -> str | None:
    """按文件名或内部 hw_env 字段查找硬件 JSON。"""
    hw_dir = os.path.join(_DATA_DIR, "hardwares")
    if not os.path.isdir(hw_dir):
        return None
    # 精确文件名匹配
    exact = os.path.join(hw_dir, f"{hw_name}.json")
    if os.path.exists(exact):
        return exact
    # 按内部 hw_env 字段搜索
    for fname in os.listdir(hw_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(hw_dir, fname)
        with open(fpath) as f:
            d = json.load(f)
        if d.get("hw_env") == hw_name:
            return fpath
    return None


@router.get("/hardwares")
async def list_hardware():
    """列出所有内置硬件（含芯片名和厂商）。"""
    hw_dir = os.path.join(_DATA_DIR, "hardwares")
    if not os.path.isdir(hw_dir):
        return []
    items = []
    for fname in os.listdir(hw_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(hw_dir, fname)) as f:
            d = json.load(f)
        name = d.get("hw_env", fname.replace(".json", ""))
        chip = d.get("chip_name", "")
        vendor = d.get("vendor", "")
        items.append({"name": name, "chip": chip, "vendor": vendor})
    return sorted(items, key=lambda x: x["name"])


@router.get("/hardwares/{name}")
async def get_hardware(name: str):
    """获取指定硬件的 JSON 定义。"""
    path = _find_hardware_file(name)
    if path is None:
        raise HTTPException(404, f"硬件 '{name}' 不存在")
    with open(path) as f:
        return json.load(f)


@router.post("/hardwares")
async def save_hardware(req: SaveHardwareRequest):
    """保存硬件规格 JSON 到本地文件。"""
    hw_dir = os.path.join(_DATA_DIR, "hardwares")
    os.makedirs(hw_dir, exist_ok=True)
    hw_name = req.hw_env or "unknown"
    path = os.path.join(hw_dir, f"{hw_name}.json")
    with open(path, "w") as f:
        json.dump(req.model_dump(), f, indent=2, ensure_ascii=False)
    logger.info("saved hardware '%s' (chip=%s)", hw_name, req.chip_name or "?")
    return {"status": "ok", "name": hw_name}


@router.delete("/hardwares/{name}")
async def delete_hardware(name: str):
    """删除指定硬件规格。"""
    path = _find_hardware_file(name)
    if path and os.path.exists(path):
        os.remove(path)
        logger.info("deleted hardware '%s'", name)
    else:
        logger.warning("delete hardware '%s': file not found", name)
    return {"status": "ok"}


# ── Modules ───────────────────────────────────────────

@router.get("/modules")
async def list_modules():
    """列出所有内置模块名称。"""
    mod_dir = os.path.join(_DATA_DIR, "modules")
    if not os.path.isdir(mod_dir):
        return []
    return sorted(
        f.replace(".json", "")
        for f in os.listdir(mod_dir)
        if f.endswith(".json")
    )


@router.get("/modules/{name}")
async def get_module(name: str):
    """获取指定模块的算子 JSON 定义。"""
    path = os.path.join(_DATA_DIR, "modules", f"{name}.json")
    if not os.path.exists(path):
        raise HTTPException(404, f"模块 '{name}' 不存在")
    with open(path) as f:
        raw = f.read().strip()
        if not raw:
            return []
        return json.loads(raw)


# ── HF Configs ───────────────────────────────────────

@router.get("/hf_configs")
async def list_hf_configs():
    """列出所有内置 HF config 名称。"""
    hf_dir = os.path.join(_DATA_DIR, "hf_configs")
    if not os.path.isdir(hf_dir):
        return []
    return sorted(
        f.replace(".json", "")
        for f in os.listdir(hf_dir)
        if f.endswith(".json")
    )


@router.get("/hf_configs/{name}")
async def get_hf_config(name: str):
    """获取指定 HF config 的 JSON 定义。"""
    path = os.path.join(_DATA_DIR, "hf_configs", f"{name}.json")
    if not os.path.exists(path):
        raise HTTPException(404, f"HF config '{name}' 不存在")
    with open(path) as f:
        return json.load(f)
