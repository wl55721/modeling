from typing import List, Dict, Type
from zrt.cost_model.base_model import BaseModel
from zrt.cost_model.model_register import ModelType, COST_MODEL_MAP
from zrt.layers.op_base import OperatorBase
from zrt.tensor_base import TensorBase

class CostModelManager:
    """成本模型管理器"""
    def __init__(self, ai_chip_config, rt_config, model_target):
        """初始化成本模型管理器"""
        self.ai_chip_config = ai_chip_config
        self.rt_config = rt_config
        self.model_target = model_target
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        """初始化模型"""
        for model_type in self.model_target:
            if model_type.name in ModelType.__members__:
                model_enum = ModelType[model_type.name]
                if model_enum in COST_MODEL_MAP:
                    model_class = COST_MODEL_MAP[model_enum]
                    self.models[model_type] = model_class(self.ai_chip_config, self.rt_config)

    def predict(self, op: OperatorBase, input_tensor: List[TensorBase], **kwargs):
        """预测算子执行时间"""
        # 尝试使用每个模型进行预测，返回第一个成功的结果
        for model_type in self.model_target:
            if model_type in self.models:
                try:
                    result = self.models[model_type](op, input_tensor, **kwargs)
                    return result.value
                except Exception as e:
                    print(f"Model {model_type.name} prediction failed: {e}")
        # 如果所有模型都失败，返回默认值
        return 0.0
