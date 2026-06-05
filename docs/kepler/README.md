# Kepler

LLM 负载建模工具

## 项目简介

Kepler 是一个 LLM 负载建模工具，支持在前端界面自由拖拽、组装算子构建模型结构，通过数学公式 CostModel 代替昂贵的真实推理或训练，快速预测 LLM 的 TTFT、TTOT、TPS、QPS 和 NPU 资源消耗。

命名寓意：正如开普勒用数学公式预测行星运动、替代了每次实际观测——Kepler 用公式替代真实 AI 计算过程，输出性能与资源消耗数据。

### 核心功能

- **手搓模型**：拖拽式算子编排，可视化构建模型计算图
- **算子库**：内置 51 个算子定义，按类别组织（Attention、Linear、MoE、Normalization 等 12 类）
- **多硬件对比**：支持同时配置多种 NPU/GPU 硬件进行横向比较
- **负载配置**：可配置 Prefill/Decode 阶段、并行策略（TP/DP/PP/EP）、量化方案

## 技术栈

- **前端**: React 19 + TypeScript + Zustand + ECharts
- **后端**: Python + FastAPI + NetworkX
- **部署**: Docker Compose 单容器

## 快速开始

```bash
# 后端
cd backend && uvicorn kepler.web.app:app --host 0.0.0.0 --port 8000

# 前端
cd frontend && npm run dev
```
