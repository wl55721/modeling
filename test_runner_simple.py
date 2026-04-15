import sys
sys.path.insert(0, 'python')

from zrt.runner.runner import Runner
from zrt.graph.graph import GlobalGraph
from zrt.graph.node import Node
from zrt.common.chip_spec import load_chip_spec

# 创建一个简单的测试图
graph = GlobalGraph()

# 创建一个 rank
rank = graph.create_rank(0)

from zrt.common.tensor_base import TensorBase, DType

# 创建输入和输出张量
input_tensor1 = TensorBase(shape=[1, 1024], dtype=DType.from_str("float16"))
input_tensor2 = TensorBase(shape=[1024, 1024], dtype=DType.from_str("float16"))
output_tensor = TensorBase(shape=[1, 1024], dtype=DType.from_str("float16"))

# 创建节点
node1 = Node(
    index=0,
    aten_op="Add",
    layer=0,
    module_path="test.module",
    component="test_component",
    stream=0,
    inputs=[input_tensor1, input_tensor1],
    outputs=[output_tensor]
)

node2 = Node(
    index=1,
    aten_op="MatMul",
    layer=0,
    module_path="test.module",
    component="test_component",
    stream=1,
    inputs=[input_tensor1, input_tensor2],
    outputs=[output_tensor]
)

# 添加节点到 rank
rank.add_op_node(node1)
rank.add_op_node(node2)

# 添加边
rank.add_op_edge(node1, node2)

# 创建芯片规格
chip_spec = load_chip_spec("H100-SXM")

# 创建 runner 并运行
runner = Runner(graph, chip_spec)
timings = runner.run()

# 使用新的结果输出功能
from zrt.utils.result_output import output_results

# 输出结果到屏幕和 Excel 文件
output_results(timings, chip_spec)
