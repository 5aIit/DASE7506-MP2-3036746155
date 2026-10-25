# 安装与复现 Release 1.0

最终入口为 `code/submission.py`；完整推理资产只有
`code/checkpoint/model.joblib` 和 `code/checkpoint/config.json`。
正式完整测试分数为 **395/467 = 84.58244111349036%**。
困难验证的 **327/355 = 92.11267605633803%** 是经过选择的开发成绩。

## 环境与数据

使用 Python 3.12；记录环境为 3.12.14、Apple M5、macOS 26.6.2。
从项目根目录执行：

```sh
cd code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_data.py
```

锁定依赖包括 NumPy 2.5.3、scikit-learn 1.9.1 和 CatBoost 1.2.8。课程原始
`data/` 已保留；不需下载其他模型或数据。为避免序列化兼容问题，应使用锁定版本。
本地 `.venv/` 可直接使用，但无需放进提交包。

## 直接评估，无需训练

```sh
python evaluate.py --split validation --reproduce
python evaluate.py --split test --reproduce
python audit.py
```

评估脚本先检查 `release.json` 的所有源文件、配置和模型哈希，再依次启动官方
`predict.py`、`score_predictions.py` 两个新进程。计时包含启动、导入、加载、
特征计算、全部融合成员和评分。BLAS 设为 1 线程，OpenMP/CatBoost 至多 4 线程。
完整资格仍须由课程指定 Xeon Platinum 8457C 上四线程、不超过 30 秒的测量确定。

结果保存在 `results/{validation,test}-predictions.json`、`*-score.json`、
`*-resources.json`。`--reproduce` 允许用未改变的版本重新生成这些文件；如需
保留原始计时，改用 `--output-dir results/recheck`，无需覆盖已有结果。
源文件或权重改变时，脚本拒绝将其当作相同发布版本重评。

`audit.py` 默认运行全部 32 项合成接口/规则测试，不读取测试标签；输出
`results/audit.json`。`release.json` 是当前发布清单；
`guide/UPSTREAM_PACKAGE_MANIFEST.json` 仅记录课程原始包，不用于核验重构代码。

## 重训与消融

```sh
python train.py
python ablate.py --output results/ablation-reproduced.json
```

`train.py` 从课程训练数据重建七个分量、三折折外预测和全部预测头，只读取冻结
配置，不使用原模型权重训练。完成拟合后才读取验证集进行精确一致性核对，不评估
测试集。新权重和结果写入 `results/retraining/`，不会覆盖正式 `checkpoint/`。
若输出目录已存在，使用 `--output results/retraining-another`；脚本拒绝覆盖。
随项目保留的本次重训结果为 `results/training.json` 和 `results/training.log`。

消融只移除一个模块，不重训、不重新搜索。七行结果应与 `results/ablation.json`
相同。历史对照与失败实验摘要见 `results/development.json`；已删除的历史搜索
实现不属于正式训练依赖。本项目保证重建最终固定配方，不宣称一条命令重做全部
自适应研究历史。

可显式使用新拟合权重预测：

```sh
python predict.py --method submission.py \
  --checkpoint results/retraining/checkpoint --split validation \
  --output results/retraining/validation-predictions.json
python score_predictions.py --split validation \
  --predictions results/retraining/validation-predictions.json \
  --output results/retraining/validation-score.json
```

## 新审计题组

按 `guide/CODE_INTERFACE_ORIGINAL.md` 的数据结构准备输入，然后运行：

```sh
python predict.py --method submission.py --checkpoint checkpoint \
  --split audit --data-dir /absolute/path/to/audit-data \
  --output results/audit-predictions.json
```

题数和 ID 任意；每题仍应有四个候选及 1024 维表示。选择器不读取 gold/reference
文件，也不重放保存的预测。评分是独立评估器的动作。

## 报告

当前仅保留英文报告源稿 `REPORT_EN.md` 和英文 PDF
`output/pdf/MP2_Report_Meng_Weitao_3036746155.pdf`；中文版报告已移除。
在项目根目录执行 `python build_report.py` 可从 Markdown 重新生成 PDF，需在
单独环境安装 ReportLab。默认嵌入 ReportLab 自带的 Bitstream Vera 字体，
无需安装中文字体。可通过 `MP2_REPORT_FONT`、`MP2_REPORT_BOLD_FONT` 指定
自定义 TrueType 字体路径。
报告工具不属于推理依赖。提交时使用同一固定版本的公开代码与 checkpoint 链接，
再通过课程网站实际创建 GitHub issue；公开仓库本身不等于已完成课程提交。
