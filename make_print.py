import os

# Name of your saved Python code file
script_filename = "rag-llm.py"  # Change this to your actual file name

# Create the LaTeX document content.
# All percent signs in the LaTeX code are doubled (%%) so they are not interpreted
# as format specifiers by Python's % operator, except for the %s placeholder.
latex_content = r"""
\documentclass[12pt]{article}
\usepackage[margin=0.5in]{geometry} %% Small margins for more text per page
\usepackage{fancyhdr}             %% For custom headers/footers (page numbers)
\usepackage{listings}             %% For formatting code
\usepackage{xcolor}               %% For color in listings
\usepackage{titling}              %% For title formatting

%% Setup page numbering (fancyhdr prints page numbers by default)
\pagestyle{fancy}
\fancyhf{}
\rfoot{\thepage}

%% Configure listings style for Python code
\lstset{
    language=Python,
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    numbers=left,
    numberstyle=\tiny\color{gray},
    keywordstyle=\color{blue},
    commentstyle=\color{green},
    stringstyle=\color{red},
    showstringspaces=false
}

\title{Retrieval Augmented Generation for Biomedical Database with AI Language Model}
\author{} %% No author
\date{}   %% No date

\begin{document}

\maketitle
\thispagestyle{empty}

\newpage
\pagestyle{fancy}  %% Reinstate page numbers after the cover page

%% Insert the code file using the listings package
\lstinputlisting{%s}

\end{document}
""" % script_filename

# Write the LaTeX content to a file
tex_filename = "document.tex"
with open(tex_filename, "w", encoding="utf-8") as f:
    f.write(latex_content)

print("LaTeX file created:", tex_filename)

