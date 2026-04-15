class RuntimeConfig:
    """运行时配置类"""
    def __init__(self):
        """初始化运行时配置"""
        self.ai_chip_config = None
        # 命令行参数
        self.model_target = []
        self.operator = "MatMul"
        self.batch_size = 32
        self.seq_len = 1024
        self.hidden_size = 768
        self.num_layers = 12
        self.num_heads = 12
        self.vocab_size = 50257
        self.policy_type = "priority"
        self.output_file = None

class AIChipConfig:
    """AI芯片配置类"""
    def __init__(self):
        """初始化AI芯片配置"""
        # 这里可以添加AI芯片的相关配置
        pass
