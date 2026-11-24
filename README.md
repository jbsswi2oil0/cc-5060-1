# 频谱涂抹式音频编辑器 (Spectral Smearing Audio Editor)

在 **时间–频率平面** 上用矩形画笔绘制增益掩码，程序在 **真实复数 STFT** 上修改
幅值（不旋转相位），再经逆变换与重叠相加（OLA）重建波形。不是在频谱图上盖颜色，
也不是用整段音量变化代替选区处理。

## 它保证了什么

| 需求 | 实现方式 | 验证 |
|------|----------|------|
| 分析窗/步长/OLA 归一化匹配 | 同一窗函数做分析与合成，按窗平方的 OLA 包络逐点归一化（伴随算子加权） | 未编辑往返误差约 `1e-15`（相对 -310 dB 以下） |
| 未编辑往返精确 | COLA 条件 + 包络归一化 | `tests/TestSTFT.test_roundtrip_*` |
| 实数音频共轭关系不被破坏 | 全程使用 `rfft/irfft`，只编辑非负频率半边，负频率半边由变换自动共轭生成 | `test_conjugate_symmetry_structural` |
| 撤销是真正回到原数值 | 保存原始信号与笔画列表，重建时从原始谱重新施加全部笔画；撤销=删除笔画后重建，**从不乘反向增益** | `test_exact_undo_returns_prior_state`（逐位相等） |
| 缩放不改变声音 | 笔画以 **秒 / Hz** 物理单位存储，按帧中心时刻与频箱中心频率栅格化；画布像素只负责显示 | `test_zoom_independence` |
| 选区边缘可解释平滑 | 高斯羽化，宽度以 σ（秒、Hz）表示；报告给出 1σ/3σ 实际作用范围 | `test_feather_is_monotone_and_sigma_scaled` |
| 局部修改不污染远处 | 未触及单元（6σ 之外）的能量泄漏量化为比值 | 所有样例泄漏 < 2e-6 |
| 导出的是真实编辑 | 写 WAV 后重新加载再分析，断言与内存重建一致 | `TestExportRoundTrip` |

## 两种跨声道策略（立体声）

实现于 `spectralpaint/engine.py`。

1. **锁相 `phase_locked`（保留声道间相位差）**
   左右声道乘以 **同一个实数增益场** `g(t,f)`。幅值改变，但逐单元的
   `arg(L) − arg(R)` 完全不变。精确谱口径下声像偏移 ≈ `1e-9 dB`、相位差变化 ≈ `1e-16 rad`。
   适合跨声道窄带处理，声像不会移动。

2. **相干区联合重估 `coherent_joint`（仅在相干区域联合）**
   用时频相干性 `γ²(t,f)=|S_LR|²/(S_LL·S_RR)`（二维平滑后的互谱/自谱）区分：
   * 相干单元（中心/共同内容，`γ² ≥ 阈值`）：用两声道平均能量匹配出一个 **公共实数增益**，
     联合重估，保持声像与相位关系；
   * 非相干单元（环境/声道独立内容）：各声道按画笔增益 **独立重估**。

> 报告同时给出 **精确谱口径**（逆变换前，反映引擎本身）与 **再分析口径**
> （OLA 重建后重新做 STFT，含选区边界的窗效应）两组声像/相位数字，避免把窗的
> 再分析泄漏误判成引擎破坏了相位。

## 谱选区表示实验（选区怎么存）

`docs/selection_representation.md`（由 `scripts/run_selection_experiment.py` 生成）
在同一逆变换原型上对比两种选区存储表示：**连续边界曲线**（秒/Hz，求值时再栅格化）
与**带原始窗参数的系数掩码**（按像素固化 / 物理重采样 / 拒绝三种重开策略）。
用同一窄带音与瞬态素材，在变窗（n_fft 2048→4096、hop 512→1024）后重开保存的选区，
以掩码边界（半增益等值线落点）与重建误差（目标带衰减、相邻带误伤、逐 click 电平）
量化取舍。结论：**选区以连续边界为唯一真实存储，掩码只作绑定窗参数的派生缓存**；
界面"应用并重建会话"（`/api/reconfigure`）即按此实现——换窗后选区不丢失，
在新网格上重新求值。复算：

```bash
.venv/bin/python scripts/run_selection_experiment.py   # 重新生成文档与全部证据
```

## 测量指标（`spectralpaint/metrics.py`）

* `pan_shift_db`：中/侧能量比导出的声像变化（dB）；
* `phase_diff_change`：声道间相位差变化（弧度，已卷绕）；
* `leakage_untouched`：6σ 之外“本不该动”的时频单元里出现的能量泄漏占比；
* `boundary_residue`：羽化环带（1σ–3σ）的能量变化；
* `time_domain`：波形时域 ΔRMS / 峰值；
* `roundtrip`：无编辑往返误差。

## 目录结构

```
spectralpaint/
  dsp.py        STFT/ISTFT、COLA 归一化、物理频率/时间坐标
  mask.py       物理单位笔画、高斯羽化、dB 复合
  selection.py  选区存储表示：连续边界曲线 / 带窗参数的系数掩码（实验对象）
  engine.py     EditSession：两种声道策略、相干性、精确撤销、重建
  metrics.py    声像/相位/泄漏/边界/往返测量
  signals.py    双音、扫频、瞬态混合、综合样例（确定性生成）
  audioio.py    WAV/FLAC/OGG 读写（soundfile）
  server.py     标准库 http.server JSON 后端 + Agg 谱图渲染
web/
  index.html    单页界面（时频画布、缩放、A/B 试听、指标面板）
  app.js        视图变换（像素↔秒/Hz）、绘制、API 交互
tests/          pytest，44 个用例
scripts/run_app.py                  启动 UI / 无头批处理
scripts/run_selection_experiment.py 谱选区表示实验（生成 docs 与 exports 证据）
scripts/ui_smoke.py                 浏览器端 UI 冒烟（Playwright 驱动真实界面）
docs/selection_representation.md    谱选区表示实验结论（由脚本生成，数字可复算）
```

## 运行

```bash
# 依赖（容器内已装入 .venv）
python -m pip install -r requirements.txt

# 1) 浏览器界面（无原生 GUI 依赖，后端只用标准库）
python scripts/run_app.py serve --port 8765
# 打开 http://127.0.0.1:8765 ，载入内置样例或上传 WAV，拖鼠标涂抹

# 2) 无头端到端批处理（生成 WAV、全部测量、导出再分析、撤销校验）
python scripts/run_app.py demo --signal mixed_sample \
    --strategy phase_locked --out exports/

# 3) 测试
python -m pytest tests -q

# 4) 浏览器 UI 冒烟（需先安装浏览器：python -m playwright install chromium）
python scripts/ui_smoke.py
```

### 界面操作

* **拖拽** 画矩形选区，坐标始终是秒/Hz；右侧读数显示物理范围。
* **滚轮** 以光标为中心缩放（Shift+滚轮只缩频率轴），**双击** 复位视图。
  缩放只改视图，不改变实际作用的频箱与声音。
* 增益（dB）、时间/频率羽化 σ 在右侧设置；`重建波形` 后可 **A/B 试听**。
* `应用并重建会话` 切换 FFT 长度/hop/窗/策略：**选区不丢失**——它们以连续
  秒/Hz 边界存储，在新分析网格上重新求值（见上方"谱选区表示实验"）。
* `撤销` 从原始谱重建回上一状态；`导出 WAV 并再分析` 落盘后重新加载校验。
* 无头环境没有音频设备时，用内置音频播放器的下载按钮保存后试听。

## 关键边界与取舍

* 增益是 **实数、非负** 的乘法，只改幅值不引入相位旋转；负增益以衰减（dB）表达。
* 羽化是高斯软边，3σ 之外仍有可见但很小的尾（约 1.9%@3σ），因此“未触及泄漏”
  默认在 6σ（尾 <1e-8）处测量，专门反映真正的谱泄漏而非预期平滑。
* 立体声像在精确谱上由锁相保证不动；重建后再分析的小幅偏移来自窗函数在选区
  边界的能量重新分配，界面与报告将两者分开列示。
* 导出为浮点 WAV（subtype=FLOAT），不做隐式归一化，以免掩盖真实增益、破坏可校验性。
