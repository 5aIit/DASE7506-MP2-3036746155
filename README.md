# DASE7506 MP2 — Release 1.0

Meng Weitao · 3036746155

本项目只保留一个正式版本：**Release 1.0**。它将原最终审查方法按功能重新整理，
算法、训练参数和模型权重不变，不再维护多个编号版本。

**结果：**完整验证 770/798（96.4912%），困难验证 327/355（92.1127%），
官方测试 395/467（84.58244111349036%）。验证集参与多轮开发选择；本次重新评估是
等价重构的复现，不是新的独立泛化实验。

## 从这里开始

- [英文报告（Markdown）](REPORT_EN.md)及[英文 PDF](output/pdf/MP2_Report_Meng_Weitao_3036746155.pdf)
- [版本与文件职责](VERSIONS.md)
- [安装、评估和重训](REPRODUCE.md)
- [审查与精简记录](CLEANUP.md)
- [AI 协助与复用披露](AI_DISCLOSURE.md)

```text
code/
  submission.py       唯一预测入口
  train.py            按固定配置从训练集重新拟合
  evaluate.py         冻结核验、官方评分与完整进程计时
  ablate.py           固定参数消融
  audit.py            完整性和合成输入回归检查
  selector/           按功能划分的特征、模型与约束
  checkpoint/         model.joblib + config.json
  tests/              两个测试文件，共 32 项检查
  results/            评分、消融、重训和历史实验摘要
  release.json        正式代码及权重的 SHA-256 清单
  data/               课程原始数据，未修改
```

`common.py`、`predict.py`、`score_predictions.py`、`check_data.py` 为未改动的课程工具。
`guide/` 保留作业要求、原始接口说明及上游包清单。旧实验编号只出现在历史证据中，
不再对应项目内的可执行版本。

快速复核：进入 `code/`，使用已安装锁定依赖的 Python 3.12 运行：

```sh
python check_data.py
python evaluate.py --split test --reproduce
python audit.py
```

本机资源记录在 `code/results/test-resources.json`；参考 Xeon CPU 的资格复核尚未完成。
正式提交请使用 Release 1.0 的固定版本，并让代码链接与 `code/checkpoint/` 链接
指向同一 Git 提交或发布标签。课程提交 issue 由提交者通过课程网站创建。
