#!/bin/bash

cat > mkdocs.yml <<EOF
site_name: TPOT2
site_url: http://epistasislab.github.io/tpot2

repo_url: https://github.com/epistasislab/tpot2
edit_uri: edit/main/source

plugins: 
  - include-markdown
  - search
  - mkdocs-jupyter:
  #     ignore_h1_titles: True
  #     execute: False
  #     include_source: True
  - mkdocstrings:
      handlers: 
        python:
          options:
            docstring_style: numpy
            show_root_full_path: False
            # show_root_toc_entry: False
  # # temp plugin
  # - exclude:
  #     glob:
  #       - tutorial/*

extra_css:
  - css/extra.css

theme:
  name: material
  features:
    # - toc.integrate
    - search.suggest
    - search.highlight
  palette:
    # light mode
    - scheme: default
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode

    # dark mode
    - scheme: slate
      toggle:
        icon: material/brightness-4
        name: Switch to light mode

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences
  - toc:
      permalink: true
  - tables
  - fenced_code
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.snippets

docs_dir: docs
site_dir: target/site

nav:
  - Home: index.md
  - Installation: installation.md
  - Using TPOT2: using.md
EOF
# static pages
echo "  - TPOT2 API:" >> mkdocs.yml
echo "    - tpot2_api/estimator.md" >> mkdocs.yml
echo "    - tpot2_api/classifier.md" >> mkdocs.yml
echo "    - tpot2_api/regressor.md" >> mkdocs.yml
echo "  - Examples:" >> mkdocs.yml
for file in docs/Tutorial/*.ipynb; do
  base=$(basename $file .ipynb)
    echo "    - Tutorial/$base.ipynb" >> mkdocs.yml
done
echo "  - Documentation:" >> mkdocs.yml
function iterate_source_files() {
  local directory="$1"

  for file in "$directory"/*; do
    if [ -f "$file" ] && [[ "$file" == *.md ]]; then
      slash_count=$(echo "$file" | grep -o '/' | wc -l)
      num_spaces=$((slash_count * 2))
      spaces=$(printf "%*s" $num_spaces)
      echo "$spaces- ${file#*/}" >> mkdocs.yml
    fi
  done

  for file in "$directory"/*; do
    if [ -d "$file" ]; then
      slash_count=$(echo "$file" | grep -o '/' | wc -l)
      num_spaces=$((slash_count * 2))
      spaces=$(printf "%*s" $num_spaces)
      last_dir=$(basename "$file")
      echo "$spaces- $last_dir:" >> mkdocs.yml
      iterate_source_files "$file"
    fi
  done
}
iterate_source_files "docs/documentation"
# make these static instead
# for file in docs/*.md; do
#   base=$(basename $file .md)
#   if [ "$base" == "index" ]; then
#     continue
#   fi
#   echo "  - $base.md" >> mkdocs.yml
# done
echo "  - contribute.md" >> mkdocs.yml
echo "  - cite.md" >> mkdocs.yml
echo "  - support.md" >> mkdocs.yml
echo "  - related.md" >> mkdocs.yml
# moved to the top
# # test docstring
# # echo "  - Tutorials:" >> mkdocs.yml
# for file in docs/tutorial/*.ipynb; do
#   base=$(basename $file .ipynb)
#     echo "    - tutorial/$base.ipynb" >> mkdocs.yml
# done
