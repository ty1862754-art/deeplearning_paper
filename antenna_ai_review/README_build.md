# 人工智能赋能天线优化设计：构建说明

## 文件

- `antenna_ai_review.tex`：论文 LaTeX 源文件
- `antenna_ai_review.pdf`：已编译完成的 PDF
- `antenna_ai_review_check.txt`：从 PDF 提取的文本，用于检查正文与参考文献位置

## 编译命令

在 `E:\code\DP_final1\antenna_ai_review` 目录下执行：

```powershell
xelatex -interaction=nonstopmode antenna_ai_review.tex
xelatex -interaction=nonstopmode antenna_ai_review.tex
```

## 页数检查

```powershell
pdfinfo antenna_ai_review.pdf
```

当前结果：

- PDF 总页数：11 页
- 参考文献从第 7 页末开始
- 正文满足“不含参考文献至少六页”的要求

## 内容处理说明

- 题目采用：`人工智能赋能天线优化设计`
- 删除了原文中的实验验证章节逻辑，没有设置“Preliminary Experiments”
- 保留并改写了原文中的方法脉络：神经网络代理模型、知识型神经网络、多保真空间映射、逆向设计、拓扑优化与生成式设计
- 保留原文作为核心参考文献，并保留了原文中的代表性引用
