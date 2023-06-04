#!/bin/bash

function iterate_files() {
    local directory="$1"
    base_dir="docs/documentation"

    for file in "$directory"/*; do
        if [ -f "$file" ] && [[ "$file" == *.py ]] && [ "$(basename "$file")" != "__init__.py" ] && \
            ! echo "$file" | grep -q "test" && [ "$(basename "$file")" != "graph_utils.py" ]; then
            directories=$base_dir/$(dirname "$file")
            file_name=$(basename "$file")
            md_file=$directories/"${file_name%.*}".md

            mkdir -p $directories && touch $md_file
            include_line=$(dirname "$file")
            include_line="${include_line//\//.}"."${file_name%.*}"
            echo "::: $include_line" > $md_file

        elif [ -d "$file" ]; then
            iterate_files "$file"
        fi
    done
}

iterate_files "tpot2"
