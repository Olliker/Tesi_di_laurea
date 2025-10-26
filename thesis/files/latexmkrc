$ENV{'TZ'} = 'Europe/Rome';
$pdflatex = 'pdflatex -shell-escape %O %S';
# Extend latexmk cleanup so -c/-C remove glossary & minted artifacts we don't keep
push @generated_exts, qw(acn acr alg glg glo gls ist lol bbl xmpi pygtex pygstyle);
