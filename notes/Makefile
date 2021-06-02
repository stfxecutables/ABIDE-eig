all: pdf

pdf:
	rm -rf _main.Rmd _main.log
	R -e "bookdown::clean_book(TRUE)"
	R -e "bookdown::render_book('notes.Rmd', 'bookdown::pdf_book', output_options = list(toc_depth=3, link_citations=TRUE),  clean=TRUE)" && cp _book/_main.pdf notes.pdf
	xdg-open notes.pdf

watch-pdf:
	watchexec --exts Rmd -i _book -i _main.Rmd --debounce 1000 make pdf

.PHONY:
	all
