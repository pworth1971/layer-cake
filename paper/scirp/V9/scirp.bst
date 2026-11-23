% scirp.bst -- reference style of Scientific Research Publishing (SCIRP) journals,
% following the reference section of paper/scirp/ijis-template.docx:
%
%   Malik, A.S., Boyko, O., Atkar, N. and Young, W.F. (2001) A Comparative Study
%   of MR Imaging Profile of Titanium Pedicle Screws. Acta Radiologica, 42, 291-293.
%   https://doi.org/...
%
% Entries are numbered in order of first citation (as unsrt). Titles are printed as
% given in the .bib file (no case change); journal and proceedings names are italic,
% volumes bold; the DOI (or URL) goes on its own line.

ENTRY
  { address author booktitle doi edition editor howpublished institution journal
    note number organization pages publisher school title type url volume year }
  {}
  { label }

INTEGERS { output.state before.all mid.sentence after.sentence after.block }
INTEGERS { nameptr namesleft numnames }
STRINGS { s t }

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}

FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}

FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}

FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ }
    { output.state after.block =
        { add.period$ write$
          newline$
          "\newblock " write$
        }
        { output.state before.all =
            'write$
            { add.period$ " " * write$ }
          if$
        }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}

FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}

FUNCTION {output.bibitem}
{ newline$
  "\bibitem{" write$
  cite$ write$
  "}" write$
  newline$
  ""
  before.all 'output.state :=
}

FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}

FUNCTION {format.link}
{ doi empty$
    { url empty$ howpublished empty$ not or
        { "" }
        { "\newline \url{" url * "}" * }
      if$
    }
    { "\newline \url{https://doi.org/" doi * "}" * }
  if$
}

FUNCTION {fin.entry}
{ add.period$
  write$
  format.link write$
  newline$
}

FUNCTION {format.names}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr "{vv~}{ll}{, jj}{, f{.}.}" format.name$ 't :=
      nameptr #1 >
        { namesleft #1 >
            { ", " * t * }
            { t "others" =
                { " et al." * }
                { " and " * t * }
              if$
            }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {format.editors}
{ editor empty$
    { "" }
    { editor format.names
      editor num.names$ #1 >
        { ", Eds." * }
        { ", Ed." * }
      if$
    }
  if$
}

FUNCTION {format.head}
{ author empty$
    { editor empty$
        { organization empty$ { "" } { organization } if$ }
        { format.editors }
      if$
    }
    { author format.names }
  if$
  's :=
  year empty$
    'skip$
    { s empty$
        { "(" year * ")" * 's := }
        { s " (" * year * ")" * 's := }
      if$
    }
  if$
  title empty$
    { s }
    { s empty$ { title } { s " " * title * } if$ }
  if$
}

INTEGERS { multiresult }

FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
    { multiresult not
      t empty$ not
      and
    }
    { t #1 #1 substring$
      duplicate$ "-" =
      swap$ duplicate$ "," =
      swap$ "+" =
      or or
        { #1 'multiresult := }
        { t #2 global.max$ substring$ 't := }
      if$
    }
  while$
  multiresult
}

FUNCTION {n.dashify}
{ 't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
        { t #1 #2 substring$ "--" = not
            { "--" *
              t #2 global.max$ substring$ 't :=
            }
            {   { t #1 #1 substring$ "-" = }
                { "-" *
                  t #2 global.max$ substring$ 't :=
                }
              while$
            }
          if$
        }
        { t #1 #1 substring$ *
          t #2 global.max$ substring$ 't :=
        }
      if$
    }
  while$
}

FUNCTION {format.pages}
{ pages empty$ { "" } { pages n.dashify } if$ }

FUNCTION {emph}
{ duplicate$ empty$ 'skip$ { "\textit{" swap$ * "}" * } if$ }

FUNCTION {bold}
{ duplicate$ empty$ 'skip$ { "\textbf{" swap$ * "}" * } if$ }

FUNCTION {article}
{ output.bibitem
  format.head output.nonnull
  new.block
  journal emph output
  volume bold output
  format.pages output
  fin.entry
}

FUNCTION {inproceedings}
{ output.bibitem
  format.head output.nonnull
  new.block
  booktitle emph output
  address output
  format.pages output
  fin.entry
}

FUNCTION {conference} { inproceedings }

FUNCTION {book}
{ output.bibitem
  format.head output.nonnull
  new.block
  edition empty$ { "" } { edition " Edition" * } if$ output
  publisher output
  address output
  fin.entry
}

FUNCTION {incollection}
{ output.bibitem
  format.head output.nonnull
  new.block
  editor empty$
    { booktitle emph }
    { "In: " format.editors * ", " * booktitle emph * }
  if$
  output
  publisher output
  address output
  format.pages output
  fin.entry
}

FUNCTION {inbook} { incollection }

FUNCTION {phdthesis}
{ output.bibitem
  format.head output.nonnull
  new.block
  "Ph.D. Thesis" output
  school output
  address output
  fin.entry
}

FUNCTION {mastersthesis}
{ output.bibitem
  format.head output.nonnull
  new.block
  "Master's Thesis" output
  school output
  address output
  fin.entry
}

FUNCTION {techreport}
{ output.bibitem
  format.head output.nonnull
  new.block
  type empty$ { "Technical Report" } { type } if$
  number empty$ 'skip$ { " " * number * } if$
  output
  institution output
  address output
  fin.entry
}

FUNCTION {misc}
{ output.bibitem
  format.head output.nonnull
  new.block
  howpublished output
  note output
  fin.entry
}

FUNCTION {manual} { misc }
FUNCTION {online} { misc }
FUNCTION {unpublished} { misc }
FUNCTION {default.type} { misc }

READ

INTEGERS { number.label longest.label.width }
STRINGS { longest.label }

FUNCTION {initialize.longest.label}
{ "" 'longest.label :=
  #1 'number.label :=
  #0 'longest.label.width :=
}

FUNCTION {longest.label.pass}
{ number.label int.to.str$ 'label :=
  number.label #1 + 'number.label :=
  label width$ longest.label.width >
    { label 'longest.label :=
      label width$ 'longest.label.width :=
    }
    'skip$
  if$
}

EXECUTE {initialize.longest.label}
ITERATE {longest.label.pass}

FUNCTION {begin.bib}
{ preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  "\begin{thebibliography}{" longest.label * "}" * write$ newline$
}

EXECUTE {begin.bib}
EXECUTE {init.state.consts}
ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
}

EXECUTE {end.bib}
