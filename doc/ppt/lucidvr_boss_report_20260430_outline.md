# LucidVR 大老板汇报草稿（2026-04-30）

文件：`doc/ppt/lucidvr_boss_report_20260430.pptx`

## 汇报主线

LucidVR 的目标是复现 FlashVSR 的高效视频超分框架，并在内部视频/图像数据上做特色适配。与 SeedVR 这类通用生成模型相比，LucidVR 的定位不是“重画视频”，而是更快、更稳定、更适合真实退化输入的 VSR 模型。

## 页结构

1. **LucidVR 总目标**：高效视频超分，面向真实退化输入。
2. **为什么需要 LucidVR**：SeedVR 生成能力强，但可能改人脸/物体，且速度成本高。
3. **最终系统形态**：数据、退化、模型、流式推理、稳定输出组成完整链路。
4. **三阶段路线**：Stage1 teacher，Stage2 sparse-causal 加速，Stage3 蒸馏。
5. **Stage1 进度**：训练链路、数据接入、退化、LQ projector 时间维对齐。
6. **数据与退化**：Yubari / Takano video / Takano image / Aliyun degradation。
7. **模型框架**：LQ video -> LQ Proj-In -> WAN DiT LoRA -> VAE decoder -> SR video。
8. **效果展示占位**：建议放两组强对比，一组好结果、一组 SeedVR 改内容案例。
9. **进度、风险与缺口**：已完成、正在推进、需要支持。
10. **Roadmap / Executive ask**：资源、固定测试集、Stage2 验证、Stage3 蒸馏。

## 需要补充的素材

- SeedVR 把人脸/物体变样的对比截图。
- FlashVSR / LucidVR 当前较好的输出截图。
- 至少一组速度数字：SeedVR 3B/7B、FlashVSR、LucidVR。
- 模型大小 / 显存 / fps。
- 如果有 GT，放 GT；没有 GT 就用 LQ / SeedVR / LucidVR 三列即可。

## 风格原则

- 不使用“我/你/我们”这类表述。
- 少讲 bug，多讲阶段目标、技术路线、当前进度、需要资源。
- 不夸大效果，使用“目标 / 当前进度 / 正在验证”的表达。
