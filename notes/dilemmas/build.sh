pandoc --standalone --mathjax --resource-path . -f markdown -t html DILEMMAS.md -s --metadata pagetitle="Eigenvalue limitations" --metadata lang="en" -H "mdstyle.html" -o DILEMMAS.html
