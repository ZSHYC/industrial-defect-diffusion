# 第 15 阶段：开源级复现规范、测试与配置化升级

## 本阶段目标

第 15 阶段不重新训练模型，也不扩展新类别。目标是把项目从“本机实验可复现”升级为“开源仓库可检查、可测试、可配置”的状态。

核心目标：

```text
1. 移除脚本中的个人电脑数据集路径默认值。
2. 增加配置文件，避免关键实验清单只写在 Python 代码里。
3. 增加 pytest 单元测试。
4. 增加 GitHub Actions CI。
5. 增加项目健康检查入口。
```

## 路径规范

所有需要 MVTec AD 的脚本都改为：

```text
优先使用 --data-root
如果未传入，则读取 DATA_ROOT 环境变量
如果仍为空，则明确报错
```

示例：

```powershell
$env:DATA_ROOT="<path-to-MVTec_AD>"
python scripts/01_explore_dataset.py --category tile
```

## 配置化升级

新增：

```text
configs/categories.json
configs/final_experiments.json
```

职责：

```text
categories.json: 维护 tile / wood / leather 的 defect types、evaluation order 和 Diffusion prompts。
final_experiments.json: 维护最终实验清单、metrics 路径、阶段时间线。
```

Python API 仍保留在：

```text
src/industrial_defect/config.py
src/industrial_defect/final_results.py
```

但配置真相源已经移动到 `configs/`。

## 测试升级

新增：

```text
tests/test_config.py
tests/test_final_results.py
tests/test_io.py
tests/test_paths.py
tests/test_reproduction_check.py
```

覆盖内容：

```text
类别配置正确性
Diffusion prompt 覆盖所有 defect types
最终实验 metrics 文件存在且字段完整
CSV / JSON 读写
DATA_ROOT 路径解析
collect-only reproduction check 行为
```

当前结果：

```text
12 passed
```

## CI 升级

新增：

```text
.github/workflows/ci.yml
```

CI 不跑训练，不下载 Diffusion 模型，只做轻量检查：

```text
py_compile
pytest
final metrics collect
dashboard generation
collect-only reproduction check
```

## 项目健康检查

新增：

```text
scripts/15_project_health_check.py
```

运行：

```powershell
python scripts/15_project_health_check.py
```

输出：

```text
outputs/final_report/project_health_check.md
```

当前检查项全部 PASS：

```text
Compile core scripts and package
Run unit tests
Collect final metrics
Generate final dashboard
Run collect-only reproduction check
Scan public files for local absolute paths
```

## 最终结论

第 15 阶段完成后，项目具备了更接近开源项目的基础质量门槛：

```text
配置可查
测试可跑
CI 可验证
路径不绑定个人电脑
最终结果可 collect-only 复现
```
