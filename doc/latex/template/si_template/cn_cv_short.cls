
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{si_template/en_cv_short}[2023/11/22 sailing-innocent modified version of cv_einstein]
\LoadClass[]{article}
\RequirePackage[rgb]{xcolor}

\RequirePackage[CJKchecksingle]{xeCJK} % XeCJK
\RequirePackage{CJKnumb} % 使用 CJK 数字-

\setCJKfamilyfont{song}[
    BoldFont=HYXinRenWenSong75W.ttf,
    ItalicFont=HYXinRenWenSong65W.ttf
]{HYXinRenWenSong55W.ttf}
\setCJKfamilyfont{hei}[
    BoldFont=HYQiHei_75S.ttf,
    ItalicFont=HYQiHei_75S.ttf
]{HYQiHei_55S.ttf}

\defaultfontfeatures{Ligatures=TeX}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip = 0pt plus 1pt minus 0.1pt
\newcommand\songti{\CJKfamily{song}} % 宋体
\newcommand\heiti{\CJKfamily{hei}}	% 黑体

\DeclareOption{theme}{
    % Have fun adjusting the hue, saturation and brightness to build your own theme
    \newcommand{\hue}{0.94} % Value between 0 and 1
    \newcommand{\saturation}{0.3}  % Value between 0 and 1
    \definecolor{darkcolor}{hsb}{\hue, \saturation, 0.20}
    \definecolor{mediumcolor}{hsb}{\hue, \saturation, 0.30}
    \definecolor{lightcolor}{hsb}{\hue, \saturation, 0.50}
    \definecolor{palecolor}{hsb}{\hue, \saturation, 0.9}
}%theme

\DeclareOption{brownish}{
    \newcommand{\hue}{0.07}
    \newcommand{\saturation}{0.3}
    \definecolor{darkcolor}{hsb}{\hue, \saturation, 0.20}
    \definecolor{mediumcolor}{hsb}{\hue, \saturation, 0.30}
    \definecolor{lightcolor}{hsb}{\hue, \saturation, 0.50}
    \definecolor{palecolor}{hsb}{\hue, \saturation, 0.9}
}%theme

\DeclareOption{mustardish}{
    \newcommand{\hue}{0.15}
    \newcommand{\saturation}{0.5}
    \definecolor{darkcolor}{hsb}{\hue, \saturation, 0.20}
    \definecolor{mediumcolor}{hsb}{\hue, \saturation, 0.30}
    \definecolor{lightcolor}{hsb}{\hue, \saturation, 0.50}
    \definecolor{palecolor}{hsb}{\hue, \saturation, 0.9}
}%theme

\DeclareOption{greenish}{
    \newcommand{\hue}{0.3}
    \newcommand{\saturation}{0.4}
    \definecolor{darkcolor}{hsb}{\hue, \saturation, 0.20}
    \definecolor{mediumcolor}{hsb}{\hue, \saturation, 0.30}
    \definecolor{lightcolor}{hsb}{\hue, \saturation, 0.50}
    \definecolor{palecolor}{hsb}{\hue, \saturation, 0.9}
}%theme

\DeclareOption{aqua}{
    \newcommand{\hue}{0.5}
    \newcommand{\saturation}{0.3}
    \definecolor{darkcolor}{hsb}{\hue, \saturation, 0.20}
    \definecolor{mediumcolor}{hsb}{\hue, \saturation, 0.30}
    \definecolor{lightcolor}{hsb}{\hue, \saturation, 0.50}
    \definecolor{palecolor}{hsb}{\hue, \saturation, 0.9}
}%theme

\DeclareOption{blueish}{
    \newcommand{\hue}{0.6}
    \newcommand{\saturation}{0.4}
    \definecolor{darkcolor}{hsb}{\hue, \saturation, 0.20}
    \definecolor{mediumcolor}{hsb}{\hue, \saturation, 0.30}
    \definecolor{lightcolor}{hsb}{\hue, \saturation, 0.50}
    \definecolor{palecolor}{hsb}{\hue, \saturation, 0.9}
}%theme

\DeclareOption{pinkish}{
    \newcommand{\hue}{0.9}
    \newcommand{\saturation}{0.3}
    \definecolor{darkcolor}{hsb}{\hue, \saturation, 0.20}
    \definecolor{mediumcolor}{hsb}{\hue, \saturation, 0.30}
    \definecolor{lightcolor}{hsb}{\hue, \saturation, 0.50}
    \definecolor{palecolor}{hsb}{\hue, \saturation, 0.9}
}%theme

\DeclareOption{burgundish}{
    \newcommand{\hue}{0.94}
    \newcommand{\saturation}{0.25}
    \definecolor{darkcolor}{hsb}{\hue, \saturation, 0.20}
    \definecolor{mediumcolor}{hsb}{\hue, \saturation, 0.30}
    \definecolor{lightcolor}{hsb}{\hue, \saturation, 0.50}
    \definecolor{palecolor}{hsb}{\hue, \saturation, 0.9}
}%theme


% Apply colors to Paracol layout
\newcommand{\paracolbackgroundoptions}{
    \backgroundcolor{c[0](4pt,4pt)(0.5\columnsep,4pt)}{lightcolor}
    \backgroundcolor{c[1](0.5\columnsep,4pt)(4pt,4pt)}[rgb]{1,1,1}
    \backgroundcolor{C[0](10000pt,10000pt)(0.5\columnsep,10000pt)}{lightcolor}
    \backgroundcolor{C[1](0.5\columnsep,10000pt)(10000pt,10000pt)}[rgb]{1,1,1}
}
%-------------------------------------------------------
\definecolor{materialpurple}{HTML}{9C27B0}
\definecolor{materialindigo}{HTML}{3F51B5}
\definecolor{materialblue}{HTML}{2196F3}
\definecolor{materialcyan}{HTML}{00BCD4}
\definecolor{materialteal}{HTML}{009688}
\definecolor{materialgreen}{HTML}{4CAF50}
\definecolor{materiallime}{HTML}{CDDC39}
\definecolor{materialamber}{HTML}{FFC107}
\definecolor{materialbrown}{HTML}{795548}
\definecolor{materialred}{HTML}{FF4436}
\definecolor{materialorange}{HTML}{FF5722}
\newcommand{\iconsize}{\huge}
\newcommand{\setasidefontcolour}{\color{white}}
% ---------------------------------------------------------------------------
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

\usepackage{si_template/cn_cv_short}
