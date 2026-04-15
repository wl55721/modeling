from typing import Dict, List
from zrt.graph.node import Node
from zrt.runner.runner import Timing
from zrt.common.chip_spec import ChipSpec
from zrt.ops.op_base import get_class_by_name

# 安装必要的库
try:
    from tabulate import tabulate
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
    from tabulate import tabulate

try:
    import pandas as pd
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd

def output_results(timings: Dict[Node, Timing], chip_spec: ChipSpec, output_file: str = "runner_results.xlsx"):
    """
    输出 runner 的运行结果到屏幕和 Excel 文件
    
    Args:
        timings: 节点的时序信息
        chip_spec: 芯片规格
        output_file: Excel 文件的路径
    """
    # 准备表格数据
    table_data = []
    for node, timing in timings.items():
        # 获取算子实例
        try:
            op_class = get_class_by_name(node.op_name)
            op = op_class(chip_spec)
            op.inputs = node.inputs
            op.outputs = node.outputs
            op_result = op.get_memory_cost()
            compute_time = op_result.total_compute_time
            memory_time = op_result.total_memory_time
            compute_flops = op_result.total_compute_flops
            memory_bytes = op_result.total_memory_bytes
        except ValueError:
            compute_time = 0.0
            memory_time = 0.0
            compute_flops = 0.0
            memory_bytes = 0.0
        
        # 准备输入和输出形状
        input_shapes = [str(t.shape) for t in node.inputs]
        output_shapes = [str(t.shape) for t in node.outputs]
        
        # 添加数据行
        table_data.append([
            node.n_id,
            node.op_name,
            ", ".join(input_shapes),
            ", ".join(output_shapes),
            compute_flops,
            compute_time,
            memory_bytes,
            memory_time,
            timing.start,
            timing.end,
            timing.duration
        ])
    
    # 计算总延迟
    total_latency = max(t.end for t in timings.values()) if timings else 0.0
    
    # 打印到屏幕
    print("Test completed successfully!")
    print(f"Total latency: {total_latency:.6f} us")
    print("\nNode and Op Information:")
    print(tabulate(table_data, headers=[
        "Node ID", "Op Name", "Input Shapes", "Output Shapes", "Compute FLOPS", 
        "Compute Time (us)", "Memory Bytes", "Memory Time (us)", 
        "Start Time (us)", "End Time (us)", "Duration (us)"
    ], tablefmt="grid"))
    
    # 输出到 CSV 文件
    df = pd.DataFrame(table_data, columns=[
        "Node ID", "Op Name", "Input Shapes", "Output Shapes", "Compute FLOPS", 
        "Compute Time (us)", "Memory Bytes", "Memory Time (us)", 
        "Start Time (us)", "End Time (us)", "Duration (us)"
    ])
    csv_file = output_file.replace(".xlsx", ".csv")
    df.to_csv(csv_file, index=False)
    print(f"\nResults exported to {csv_file}")
