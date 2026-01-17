update.packages(ask = FALSE, checkBuilt = TRUE)
install.packages("tinytex")
install.packages("knitr", dep = TRUE)
install.packages(c("rmarkdown", "ggplot2", "kableExtra", "gridExtra", "reshape2", "patchwork"))
update.packages(ask = FALSE, checkBuilt = TRUE)
tinytex::tlmgr_update()
tinytex::tlmgr_install(c("amsmath","amssymb", "amsfonts","geometry","hyperref","fancyvrb","wrapfig","parskip","upquote","microtype","xcolor","graphicx","ulem","caption"))
tinytex::tlmgr_install(c("environ","trimspaces", "etoolbox", "latex-amsmath-dev","caption", "float","subcaption"))
tinytex::tlmgr_install(c("booktabs","microtype","geometry","hyperref","xcolor"))
tinytex::tlmgr_install(c("natbib", "biblatex", "biber"))
tinytex::tlmgr_install(c("xfrac","lineno"))
tinytex::tlmgr_install(c("l3packages", "l3kernel", "l3backend"))
tinytex::tlmgr_install(c("times", "multirow", "array"))
tinytex::tlmgr_install(c(
  "tex-gyre",      # Alternative to times font
  "tex-gyre-math",
  "helvetic",      # Alternative fonts
  "courier",
  "mathptmx",      # Times font for LaTeX
  "psnfss",        # PostScript fonts
  "graphics", 
  "graphicx",
  "float",
  "placeins",
  "sectsty",
  "titlesec"
))
tinytex::tlmgr_install(c("grfext", "graphics-def", "graphics-cfg", "graphics"))
tinytex::tlmgr_update("graphics")
# tinytex::tlmgr(c("install", "--reinstall", "graphics"))
tinytex::tlmgr_install(c(
  "cm-super",
  "cm-super-x2",
  "tex-gyre",
  "tex-gyre-math",
  "cm",
  "amsfonts",
  "lm",
  "lmodern",
  "ec",
  "latex-fonts",
  "fontname",
  "fontspec",
  "fontaxes",
  "fontenc"
))
tinytex::tlmgr_install("texlive-fonts-recommended")
tinytex::tlmgr_install("texlive-font-utils")