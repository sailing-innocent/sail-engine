\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{si_template/cn_note}[2023/08/15 Class for sailing-innocent's Personal Chinese Notes]

\RequirePackage{kvoptions}
\RequirePackage{etoolbox}
% Declaration of options
% ----------------------
\SetupKeyvalOptions{
    family = @SI,
    prefix = @SI@
}
\DeclareStringOption[si_template/default_cover]{coverpage} % 封面页
\newcommand{\covertext}{Title}

\DeclareStringOption[a4]{geometry}	% 版面 (默认 a4)
\PassOptionsToPackage{dvipsnames}{xcolor}	% 让 xcolor 带 dvipsnames 选
% Execution of options
% --------------------
\ProcessKeyvalOptions*
\relax

% Package Loading
% ---------------
% based on article class
\LoadClass{article}
\RequirePackage{fontspec} % XeLaTeX
\RequirePackage[CJKchecksingle]{xeCJK} % XeCJK
\RequirePackage{CJKnumb} % 使用 CJK 数字
% 引入 AMS 宏包 + mathtools
\RequirePackage[intlimits]{amsmath}
\RequirePackage{amssymb}
\RequirePackage[centercolon]{mathtools}

% hyperref
\RequirePackage{hyperref}

% 支持直接引入 PDF 页面
\RequirePackage{pdfpages}

% 加入字串处理功能
\RequirePackage{xstring}

\RequirePackage{emptypage}
\RequirePackage[many]{tcolorbox}		% 制作方框

\RequirePackage{setspace}	% 设定适于中文排版的行距

% TIKZ & PGFPLOTS
\RequirePackage{tikz}
\usetikzlibrary{shapes.symbols}	% 稍后定义 hint 环境所需
\usetikzlibrary{3d, math}
\RequirePackage{pgfplots}
\RequirePackage{pgfplotstable}

% 载入 paralist
\RequirePackage{paralist}

% control figure place
\RequirePackage{float}
\RequirePackage{geometry}

% add figure
\RequirePackage{graphicx}
\RequirePackage{subcaption}

% for color
\RequirePackage{soul}
\RequirePackage{color}
\RequirePackage{xcolor}

\RequirePackage{colortbl} % 表格颜色
\RequirePackage{listings} % 代码块

% 设置天眉所需
\RequirePackage{fancyhdr}
% 使空页恒空
\fancypagestyle{plain}{
    \fancyhead{}
    \renewcommand{\headrulewidth}{0pt}
}

% Main codes
% ----------
\usetikzlibrary{shapes.symbols}	% 稍后定义 hint 环境所需
\usetikzlibrary{3d}

% 定义在 draftmark=true 模式下显示版本信息的指令
\RequirePackage[iso, english]{isodate}  % 使 \today 印出 yyyy-mm-dd

\onehalfspacing	% 行距
%\raggedbottom	% 减小页面空白
\setlength{\parindent}{2em}	% 设置适合于汉语排版的段落缩进
% 汉化图表引用
\renewcommand{\figurename}{图}
\renewcommand{\tablename}{表}
\renewcommand{\listfigurename}{图片索引}
\renewcommand{\listtablename}{表格索引}

% 输入字体设置
% 设置 xeCJK 字体及中文数字
\setCJKmainfont[
    BoldFont=FandolSong-Bold.otf,
    ItalicFont=FandolKai-Regular.otf
]{FandolSong-Regular.otf}
\setCJKsansfont[
    BoldFont=FandolHei-Bold.otf
]{FandolHei-Regular.otf}
\setCJKmonofont[
    BoldFont=FandolHei-Bold.otf,
]{FandolHei-Regular.otf}
\setCJKfamilyfont{kai}[
    BoldFont=FandolKai-Regular.otf, ItalicFont=FandolKai-Regular.otf
]{FandolKai-Regular.otf}
\setCJKfamilyfont{song}[
    BoldFont=FandolSong-Bold.otf,
    ItalicFont=FandolKai-Regular.otf
]{FandolSong-Regular.otf}

\setCJKfamilyfont{fangsong}[
    BoldFont=FandolSong-Bold.otf,
    ItalicFont=FandolKai-Regular.otf
]{FandolFang-Regular.otf}
\setCJKfamilyfont{hei}[
    BoldFont=FandolHei-Bold.otf,
    ItalicFont=FandolHei-Regular.otf
]{FandolHei-Regular.otf}

\setCJKfamilyfont{hei2}{Noto Sans CJK SC}
\setCJKfamilyfont{sectionfont}[
    BoldFont=* Black
]{Noto Sans CJK SC}

\setCJKfamilyfont{pffont}[
    BoldFont=* Medium
]{Noto Sans CJK SC}	% 证明用的字体
\setCJKfamilyfont{emfont}[
    BoldFont=FandolHei-Regular.otf
]{FandolHei-Regular.otf}	% 强调用的字体

\defaultfontfeatures{Ligatures=TeX}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip = 0pt plus 1pt minus 0.1pt

% 以下设置字体相关命令, 用于定理等环境中.
\newcommand\kaishu{\CJKfamily{kai}} % 楷体
\newcommand\songti{\CJKfamily{song}} % 宋体
\newcommand\heiti{\CJKfamily{hei}}	% 黑体
\newcommand\fangsong{\CJKfamily{fangsong}} % 仿宋
\renewcommand{\em}{\bfseries\CJKfamily{emfont}} % 强调

\AtEndPreamble{
    \RequirePackage[thmmarks, amsmath, hyperref]{ntheorem}	% 设置定理环境所需
    % 若 hyperref 已载入, 则按 colors 的 Bool 值设置链接色彩.
    \@ifpackageloaded{hyperref}{\hypersetup{hidelinks}}
    {}
    % 设置页面尺寸
    \RequirePackage{geometry}
    \IfStrEq{\@SI@geometry}{a4}{% 载入 a4 版面设置
        \geometry{
            paper=a4paper,
            top=3cm,
            inner=2.54cm,
            outer=2.54cm,
            bottom=3cm,
            headheight=6ex,
            headsep=6ex,
            twoside,
            asymmetric
        }}{}

    % 设置天眉所需
    \RequirePackage{fancyhdr}
    % 使空页恒空
    \fancypagestyle{plain}{
        \fancyhead{}
        \renewcommand{\headrulewidth}{0pt}
    }

    % 设置天眉
    \pagestyle{fancy}
    \renewcommand{\sectionmark}[1]{\markright{\S\arabic{section} \quad #1}}
    \fancyhf{} % 先清空
    \fancyhead[EC]{\CJKfamily{hei2}\footnotesize{\leftmark}\vspace{1mm}}
    \fancyhead[OC]{\CJKfamily{hei2}\footnotesize{\rightmark}\vspace{1mm}}
    \fancyhead[LE,RO]{{\footnotesize \thepage}\vspace{1mm}} %
    \fancyhead[RE,LO]{}
    \renewcommand{\headrulewidth}{0pt}	% 天眉暂不加横线
    \renewcommand{\footrulewidth}{0pt}
    \addtolength{\headheight}{0.5pt}
}

\AtBeginDocument{}
\AtEndDocument{}