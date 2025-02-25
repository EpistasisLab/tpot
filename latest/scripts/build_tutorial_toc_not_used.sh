#!/bin/bash

for file in docs/tutorial/*.html; do
    base=$(basename "$file" .html)
    echo "<div><embed width=\"100%\" height=\"800\" src=\"../$base.html\" /></div>" > "docs/tutorial/$base.md"
done
