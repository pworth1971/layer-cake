% ijis.cls -- layout of the SCIRP International Journal of Intelligence Science
% (IJIS) author template (paper/scirp/ijis-template.docx), rebuilt for LaTeX.
%
% Page 21.0 x 27.5 cm; margins 1.5 cm (sides) and 2.5 cm (top/bottom); the body
% text runs in a 12.71 cm column indented 5.29 cm from the left margin, which leaves
% a side column used on page 1 for the citation, dates and licence. Body text is
% 10 pt on an exact 15 pt line. Headings, rules and caption labels use the template
% accent colour #943634.
%
% Fonts: the template asks for Minion Pro Capt (body), Cambria (headings) and
% Calibri (journal furniture). With pdfLaTeX the class uses the free, closely
% matched Crimson Pro, Caladea (metric-compatible with Cambria) and Carlito
% (metric-compatible with Calibri), so the source builds with a standard TeX Live.
% With XeLaTeX it uses the template's own fonts when they are installed.

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ijis}[2026/10/23 IJIS (SCIRP) article layout]
\LoadClass[10pt]{article}

% ---------------------------------------------------------------- fonts
\RequirePackage{iftex}
\RequirePackage{amsmath,amssymb}
\ifPDFTeX
  % pdfLaTeX: free equivalents of the template fonts (all in TeX Live)
  \RequirePackage[T1]{fontenc}
  \RequirePackage[lining]{CrimsonPro}
  \RequirePackage[varqu,varl]{inconsolata}
  \RequirePackage{carlito}
  \RequirePackage[cmintegrals]{newtxmath}
  \newcommand*{\headfont}{\fontfamily{Caladea-TLF}\selectfont}
  \newcommand*{\figfont}{\fontfamily{ptm}\fontsize{7.5}{9}\selectfont}
  \RequirePackage[protrusion=true,expansion=true]{microtype}
\else
  % XeLaTeX / LuaLaTeX: the template's own fonts where they are installed
  % (Minion Pro, Cambria, Calibri), otherwise the free equivalents.
  \RequirePackage[cmintegrals]{newtxmath}
  \RequirePackage[no-math]{fontspec}
  \IfFontExistsTF{Minion Pro Capt}{\setmainfont{Minion Pro Capt}}
    {\IfFontExistsTF{Minion Pro}{\setmainfont{Minion Pro}}{\setmainfont{Crimson Pro}[Numbers=Lining]}}
  \IfFontExistsTF{Calibri}{\setsansfont{Calibri}}{\setsansfont{Carlito}}
  \IfFontExistsTF{Cambria}{\newfontfamily\ijis@head{Cambria}}{\newfontfamily\ijis@head{Caladea}}
  \newcommand*{\headfont}{\ijis@head}
  \IfFontExistsTF{Times New Roman}{\newfontfamily\ijis@fig{Times New Roman}}{\newfontfamily\ijis@fig{TeX Gyre Termes}}
  \newcommand*{\figfont}{\ijis@fig\fontsize{7.5}{9}\selectfont}
  \RequirePackage[protrusion=true]{microtype}
\fi
\linespread{1.25}% 10 pt type on an exact 15 pt line, as in the template
\setlength{\emergencystretch}{1.5em}

% ---------------------------------------------------------------- page
\RequirePackage[paperwidth=21cm, paperheight=27.5cm,
  left=6.79cm, right=1.5cm, top=2.5cm, bottom=2.5cm,
  headheight=14pt, headsep=0.55cm, footskip=1.05cm]{geometry}
\newlength{\ijisoffset}\setlength{\ijisoffset}{5.29cm}   % side column width
\newlength{\ijisfullwidth}% both set in \maketitle, after geometry has fixed \textwidth;
\newlength{\ijisbodywidth}% \textwidth itself is redefined inside minipages

\setlength{\parindent}{1em}
\setlength{\parskip}{0pt}
\RequirePackage[hang,flushmargin]{footmisc}
\renewcommand{\footnotelayout}{\linespread{1.05}\selectfont}

% ---------------------------------------------------------------- colour, links
\RequirePackage{xcolor}
\definecolor{ijisred}{HTML}{943634}
\definecolor{ijisblue}{HTML}{0000FF}
\definecolor{ijisgrey}{HTML}{7F7F7F}
\RequirePackage{graphicx}
\RequirePackage{etoolbox}
\RequirePackage[numbers,square,sort&compress]{natbib}
\RequirePackage[colorlinks=true, linkcolor=black, citecolor=black, urlcolor=ijisblue]{hyperref}
\RequirePackage{xurl}% long URLs in the reference list may break anywhere
\urlstyle{same}

% ---------------------------------------------------------------- headings
\RequirePackage{titlesec}
\titleformat{\section}{\headfont\fontsize{12}{15}\bfseries\color{ijisred}}{\thesection.}{0.35em}{}
\titleformat{\subsection}{\headfont\fontsize{11}{15}\bfseries\color{ijisred}}{\thesubsection.}{0.35em}{}
\titleformat{\subsubsection}{\headfont\fontsize{10}{15}\bfseries\color{ijisred}}{\thesubsubsection.}{0.35em}{}
\titleformat{\paragraph}[runin]{\normalfont\bfseries}{}{0pt}{}
\titlespacing*{\section}{0pt}{12pt}{6pt}
\titlespacing*{\subsection}{0pt}{8pt}{4pt}
\titlespacing*{\subsubsection}{0pt}{6pt}{3pt}
\titlespacing*{\paragraph}{0pt}{4pt}{0.6em}

% ---------------------------------------------------------------- lists
\RequirePackage{enumitem}
\setlist{topsep=2pt, itemsep=1pt, parsep=0pt}
\setlist[itemize]{label=\textbullet, leftmargin=1.4em}
\setlist[enumerate]{leftmargin=1.6em}

% ---------------------------------------------------------------- floats
\RequirePackage{booktabs,colortbl,tabularx,multirow,float}
\RequirePackage{changepage}
\setlength{\heavyrulewidth}{0.6pt}
\setlength{\lightrulewidth}{0.3pt}
\arrayrulecolor{ijisred!70}
\RequirePackage[font=small, labelfont={bf,color=ijisred}, labelsep=period,
  justification=raggedright, singlelinecheck=false, skip=5pt]{caption}
\captionsetup[table]{position=top}
\AtBeginEnvironment{table}{\linespread{1}\selectfont}
\AtBeginEnvironment{figure}{\linespread{1}\selectfont}
% Oversized floats may use the full width, including the side column.
\newenvironment{fullwidth}{\begin{adjustwidth}{-\ijisoffset}{0pt}}{\end{adjustwidth}}

% Cross-references in the template style: "Figure 1", "Table 1" in bold accent colour.
\newcommand{\figref}[1]{\textcolor{ijisred}{\textbf{Figure~\ref{#1}}}}
\newcommand{\tabref}[1]{\textcolor{ijisred}{\textbf{Table~\ref{#1}}}}
\newcommand{\figsref}[2]{\textcolor{ijisred}{\textbf{Figures~\ref{#1} and~\ref{#2}}}}
\newcommand{\tabsref}[2]{\textcolor{ijisred}{\textbf{Tables~\ref{#1} and~\ref{#2}}}}

% ---------------------------------------------------------------- bibliography
\renewcommand{\bibnumfmt}[1]{#1.}
\setlength{\bibsep}{3.6pt}
\renewcommand{\bibfont}{\small\linespread{1.08}\selectfont}
\setlength{\bibhang}{2em}
\renewcommand{\refname}{References}

% ---------------------------------------------------------------- front matter data
\newcommand{\ijis@runauthors}{}
\newcommand{\runningauthors}[1]{\renewcommand{\ijis@runauthors}{#1}}
\newcommand{\ijis@affiliation}{}
\newcommand{\affiliation}[1]{\renewcommand{\ijis@affiliation}{#1}}
\newcommand{\ijis@email}{}
\newcommand{\email}[1]{\renewcommand{\ijis@email}{#1}}
\newcommand{\ijis@titlenote}{}
\newcommand{\titlenote}[1]{\renewcommand{\ijis@titlenote}{#1}}
\newcommand{\ijis@howtocite}{}
\newcommand{\howtocite}[1]{\renewcommand{\ijis@howtocite}{#1}}
\newcommand{\ijis@abstract}{}
\newcommand{\abstracttext}[1]{\renewcommand{\ijis@abstract}{#1}}
\newcommand{\ijis@keywords}{}
\newcommand{\keywords}[1]{\renewcommand{\ijis@keywords}{#1}}
\newcommand{\ijis@year}{2026}
\newcommand{\ijisyear}[1]{\renewcommand{\ijis@year}{#1}}
\newcommand{\ijis@volume}{*(*), *-*}
\newcommand{\ijisvolume}[1]{\renewcommand{\ijis@volume}{#1}}
\newcommand{\ijis@doi}{10.4236/ijis.2026.*****}
\newcommand{\ijisdoi}[1]{\renewcommand{\ijis@doi}{#1}}
\newcommand{\ijis@received}{**** **, 2026}
\newcommand{\ijis@accepted}{**** **, 2026}
\newcommand{\ijis@published}{**** **, 2026}
\newcommand{\received}[1]{\renewcommand{\ijis@received}{#1}}
\newcommand{\accepted}[1]{\renewcommand{\ijis@accepted}{#1}}
\newcommand{\published}[1]{\renewcommand{\ijis@published}{#1}}
\newcommand{\ijis@license}{This work is licensed under the Creative Commons Attribution International License (CC BY 4.0).\\
  \url{http://creativecommons.org/licenses/by/4.0/}}
\newcommand{\license}[1]{\renewcommand{\ijis@license}{#1}}
\newcommand{\ijis@ccbadge}{\includegraphics[height=0.52cm]{ijis/cc-by.png}}
\newcommand{\ccbadge}[1]{\renewcommand{\ijis@ccbadge}{#1}}

% ---------------------------------------------------------------- headers and footers
\RequirePackage{fancyhdr}
\newcommand{\ijis@journal}{International Journal of Intelligence Science}
\newcommand{\ijis@doifoot}{\textcolor{ijisred}{\textbf{DOI: \ijis@doi}}}
\fancypagestyle{ijisfirst}{%
  \fancyhf{}\fancyhfoffset[L]{\ijisoffset}%
  \renewcommand{\headrulewidth}{0pt}%
  \fancyhead[L]{\smash{\raisebox{-0.3cm}{\includegraphics[height=1.25cm]{ijis/scirp-logo.png}}}}%
  \fancyhead[R]{\smash{\raisebox{-0.3cm}{\parbox[b]{9cm}{\raggedleft\sffamily\bfseries\scriptsize\linespread{1.15}\selectfont
    \ijis@journal, \ijis@year, \ijis@volume\\
    \href{https://www.scirp.org/journal/ijis}{\underline{https://www.scirp.org/journal/ijis}}\\
    ISSN Online: 2163-0356\\ ISSN Print: 2163-0283}}}}%
  \renewcommand{\footrulewidth}{0.8pt}%
  \fancyfoot[L]{\sffamily\footnotesize\ijis@doifoot\qquad\textcolor{ijisred}{\textbf{\ijis@published}}}%
  \fancyfoot[C]{\sffamily\footnotesize\textbf{\thepage}}%
  \fancyfoot[R]{\sffamily\footnotesize\textcolor{ijisred}{\textbf{\ijis@journal}}}}
\fancypagestyle{ijis}{%
  \fancyhf{}\fancyhfoffset[L]{\ijisoffset}%
  \renewcommand{\headrulewidth}{1.4pt}%
  \renewcommand{\headrule}{{\color{ijisgrey}\hrule height\headrulewidth width\headwidth \vskip-\headrulewidth}}%
  \renewcommand{\footrulewidth}{0.8pt}%
  \fancyhead[L]{\sffamily\footnotesize\bfseries\ijis@runauthors}%
  \fancyfoot[L]{\sffamily\footnotesize\ijis@doifoot}%
  \fancyfoot[C]{\sffamily\footnotesize\textbf{\thepage}}%
  \fancyfoot[R]{\sffamily\footnotesize\textcolor{ijisred}{\textbf{\ijis@journal}}}}
\pagestyle{ijis}

% ---------------------------------------------------------------- title page
\renewcommand{\maketitle}{%
  \setlength{\ijisbodywidth}{\textwidth}%
  \setlength{\ijisfullwidth}{\dimexpr\textwidth+\ijisoffset\relax}%
  \thispagestyle{ijisfirst}%
  \noindent\hspace*{-\ijisoffset}%
  \begin{minipage}[t]{\ijisfullwidth}
    \vspace*{0.1cm}%
    {\raggedright\headfont\fontsize{20}{23}\selectfont\bfseries \@title%
      \ifx\ijis@titlenote\@empty\else\textsuperscript{*}\fi\par}
    \vspace{0.35cm}%
    {\fontsize{11}{15}\selectfont\bfseries \@author\par}
    \vspace{0.15cm}%
    {\footnotesize\linespread{1.1}\selectfont \ijis@affiliation\\ Email: \ijis@email\par}
    \vspace{0.35cm}%
    \begin{minipage}[t]{\dimexpr\ijisoffset-0.45cm\relax}
      \vspace*{0.55cm}%
      \scriptsize\linespread{1.15}\selectfont\raggedright
      \textbf{How to cite this paper:} \ijis@howtocite\par\medskip
      \textbf{Received:} \ijis@received\\
      \textbf{Accepted:} \ijis@accepted\\
      \textbf{Published:} \ijis@published\par\medskip
      Copyright \copyright\ \ijis@year\ by author(s) and Scientific Research Publishing Inc.
      \ijis@license\par\smallskip
      \ijis@ccbadge\hspace{0.25cm}\includegraphics[height=0.52cm]{ijis/open-access.png}\par
    \end{minipage}\hfill
    \begin{minipage}[t]{\ijisbodywidth}
      {\color{ijisred}\rule{\linewidth}{1.5pt}}\par\vspace{4pt}
      {\headfont\fontsize{12.5}{15}\selectfont\bfseries\color{ijisred}Abstract\par}\vspace{3pt}
      \ijis@abstract\par\vspace{4pt}
      {\headfont\fontsize{12.5}{15}\selectfont\bfseries\color{ijisred}Keywords\par}\vspace{3pt}
      \raggedright\ijis@keywords\par\vspace{6pt}
      {\color{ijisred}\rule{\linewidth}{1.5pt}}
    \end{minipage}
  \end{minipage}\par
  \ifx\ijis@titlenote\@empty\else
    \begingroup\renewcommand{\thefootnote}{}\footnotetext{\textsuperscript{*}\ijis@titlenote}\endgroup
  \fi
  \vspace{0.6cm}}

% Component heads (Acknowledgements, Author Contributions, Conflicts of Interest,
% References) are unnumbered headings in the section style.
\newcommand{\componenthead}[1]{\section*{#1}}

\endinput
