

from enum import Enum, auto
from typing import Dict, Type
from zrt.cost_model.base_model import BaseModel

class ModelType(Enum):
    LOOKUP = auto()
    TILESIM_ENGI = auto()
    TILESIM_THEO = auto()
    TILESIM_ENGI_DSL = auto()
    THEO_MODEL = auto()

COST_MODEL_MAP: Dict[ModelType, Type[BaseModel]] = {}

def register_model():
    from zrt.cost_model.lookup_table import LookupTableModel
    try:
        from zrt.cost_model.tilesim.tilesim_adapter import TilesimEngDSLModel, TilesimTheoModel, TilesimEngModel
    except ImportError as e:
        raise RuntimeError(f"Failed to import tilesim models: {e}")
    from zrt.cost_model.theo_model.theo_model import TheoreticalModel        
    
    COST_MODEL_MAP.update({
        ModelType.LOOKUP: LookupTableModel,
        ModelType.TILESIM_ENGI: TilesimEngModel,
        ModelType.TILESIM_THEO: TilesimTheoModel,
        ModelType.TILESIM_ENGI_DSL: TilesimEngDSLModel,
        ModelType.THEO_MODEL: TheoreticalModel,
    })