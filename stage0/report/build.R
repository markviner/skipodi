#!/usr/bin/env Rscript
require("rmarkdown");

render(
    "main.Rmd",
    output_format = "pdf_document",
    output_file = "report.pdf",
    output_dir = "output/",
    run_pandoc = TRUE,
    encoding = "UTF-8"
);
