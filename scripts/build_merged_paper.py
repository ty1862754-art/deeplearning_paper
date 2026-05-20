from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "文献"
OUT = ROOT / "merged_paper_tex"

MAIN_FILES = [
    ("摘要", SRC / "摘要.md"),
    ("引言", SRC / "第一章引言.md"),
    ("文献方法综述", SRC / "第二章文献方法综述.md"),
    ("讨论与分析", SRC / "第三章讨论与分析.md"),
    ("改进与未来展望", SRC / "第四章未来展望.md"),
    ("总结", SRC / "第五章总结.md"),
]

APPENDIX_FILES = [
    ("天线优化设计相关论文内容提取", SRC / "天线优化设计论文内容提取.md"),
]


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    text = text.replace("\ue5cf", "").replace("\ue5d2", "")
    escaped = "".join(replacements.get(ch, ch) for ch in text)
    escaped = re.sub(r"https?://[^\s]+", lambda m: r"\url{" + m.group(0).replace(r"\_", "_") + "}", escaped)
    return escaped


def clean_heading(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\d+(?:\.\d+)*\s*", "", text)
    return text


def convert_inline(text: str) -> str:
    text = text.strip()
    return tex_escape(text)


def md_to_tex(md: str, default_section: str | None = None) -> str:
    lines = md.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    saw_section = False
    in_itemize = False
    in_quote = False

    def close_lists() -> None:
        nonlocal in_itemize, in_quote
        if in_itemize:
            out.append(r"\end{itemize}")
            in_itemize = False
        if in_quote:
            out.append(r"\end{quote}")
            in_quote = False

    if default_section:
        out.append(r"\section{" + tex_escape(default_section) + "}")
        saw_section = True

    for raw in lines:
        line = raw.rstrip()
        if not line:
            close_lists()
            out.append("")
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            close_lists()
            level = len(heading.group(1))
            title = tex_escape(clean_heading(heading.group(2)))
            if level == 1 and not saw_section:
                out.append(r"\section{" + title + "}")
                saw_section = True
            elif level == 1:
                out.append(r"\section{" + title + "}")
            elif level == 2:
                out.append(r"\subsection{" + title + "}")
            elif level == 3:
                out.append(r"\subsubsection{" + title + "}")
            else:
                out.append(r"\paragraph{" + title + "}")
            continue

        quote = re.match(r"^>\s?(.*)$", line)
        if quote:
            if in_itemize:
                out.append(r"\end{itemize}")
                in_itemize = False
            if not in_quote:
                out.append(r"\begin{quote}")
                in_quote = True
            out.append(convert_inline(quote.group(1)) + r"\\")
            continue

        item = re.match(r"^\s*[-*]\s+(.+)$", line)
        if item:
            if in_quote:
                out.append(r"\end{quote}")
                in_quote = False
            if not in_itemize:
                out.append(r"\begin{itemize}")
                in_itemize = True
            out.append(r"\item " + convert_inline(item.group(1)))
            continue

        close_lists()
        out.append(convert_inline(line) + "\n")

    close_lists()
    return "\n".join(out)


def make_merged_md() -> str:
    chunks: list[str] = []
    for title, path in MAIN_FILES:
        chunks.append(f"# {title}\n\n")
        chunks.append(path.read_text(encoding="utf-8").strip())
        chunks.append("\n\n")
    for title, path in APPENDIX_FILES:
        chunks.append(f"# 附录：{title}\n\n")
        chunks.append(path.read_text(encoding="utf-8").strip())
        chunks.append("\n\n")
    return "".join(chunks)


def make_tex() -> str:
    body: list[str] = []
    abstract = MAIN_FILES[0][1].read_text(encoding="utf-8")
    abstract = abstract.replace("摘要：", "").replace("关键词：", r"\par\noindent\textbf{关键词：}")
    body.append(r"\begin{abstract}")
    body.append(md_to_tex(abstract))
    body.append(r"\end{abstract}")

    for title, path in MAIN_FILES[1:]:
        md = path.read_text(encoding="utf-8")
        body.append(md_to_tex(md, default_section=title if not md.lstrip().startswith("#") else None))

    body.append(r"\appendix")
    body.append(r"\small")
    for title, path in APPENDIX_FILES:
        body.append(md_to_tex(path.read_text(encoding="utf-8"), default_section=title))

    return r"""\documentclass[UTF8,a4paper,12pt]{ctexart}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{url}
\usepackage{enumitem}
\usepackage{setspace}
\usepackage{titlesec}
\usepackage{amssymb}
\usepackage{newunicodechar}
\newunicodechar{ψ}{$\psi$}
\newunicodechar{□}{$\square$}
\geometry{left=2.8cm,right=2.8cm,top=2.6cm,bottom=2.6cm}
\hypersetup{colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue}
\setstretch{1.35}
\setlist{nosep}
\title{基于人工智能的天线优化设计研究综述}
\author{}
\date{\today}
\begin{document}
\maketitle
\tableofcontents
\newpage
""" + "\n".join(body) + "\n" + r"\end{document}" + "\n"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "merged_paper.md").write_text(make_merged_md(), encoding="utf-8")
    (OUT / "paper.tex").write_text(make_tex(), encoding="utf-8")
    bib = ROOT / "antenna_ai_review" / "references.bib"
    if bib.exists():
        shutil.copy2(bib, OUT / "references.bib")


if __name__ == "__main__":
    main()
