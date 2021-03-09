#!/usr/bin/env bash

# Dependencies:
# 1. inkscape
# 2. pdfunite
# 3. (optinal) minify (https://github.com/tdewolff/minify/tree/master/cmd/minify)
#    `sudo apt install minify`

# Assumptions:
# 1. figures are made in SVG format
# 2. figure files are inside a "svg" directory under ${ROOT_DIR}
# 3. figures are consecutively and consistently labeled
# 3a. Main figures are named Figure<X>.svg and supplementary as FigureS<X>.svg,
#     where <X> is a number (multiple digits allowed).
# 4. (optional) Main figures have a "Figure <X>" and Supplementary Figures a
#    "Supplementary Figure <X>" SVG text label where <X> is a number (multiple digits allowed).

# Editing tips:
# 1. Remove clones and most non-essential grouping from the SVG
# 2. Rasterize large elements such as heatmaps, swarmplots, etc and export them as 300dpi png
# 2.b Rasterization reduces file size and memory usage when editing but one drawback is that it cannot be further reduced.
# 2.c Sometimes it's better not to rasterize but imediately minify a SVG.


echo "Preparing manuscript figures"

ROOT_DIR=`pwd`
readarray MAIN_FIGURES < <( \
    find svg \
        -maxdepth 1 \
        -regextype posix-extended -regex '.*Figure[[:digit:]]+\.svg' | \
    sort)
readarray SUPP_FIGURES < <( \
    find svg \
        -maxdepth 1 \
        -regextype posix-extended -regex '.*FigureS[[:digit:]]+\.svg' | \
    sort)
FIGURES=( "${MAIN_FIGURES[@]}" "${SUPP_FIGURES[@]}" )
NUMBER_MAIN_FIGURES=${#MAIN_FIGURES[@]}
NUMBER_SUPP_FIGURES=${#SUPP_FIGURES[@]}
CURRENT_DATE=$(date '+%Y%m%d')
# CURRENT_DATE="20210130"
# CURRENT_DATE="final"
MINIFY="TRUE"  # whether to use SVG minification
CLEANUP_TEMP="TRUE"
DPI=200

echo "Working in '$ROOT_DIR' directory."
echo -e "Found ${NUMBER_MAIN_FIGURES} main figures: \n ${MAIN_FIGURES[@]}"
echo -e "Found ${NUMBER_SUPP_FIGURES} supplementary figures: \n ${SUPP_FIGURES[@]}"


cd $ROOT_DIR
mkdir -p {svg/minified,pdf,png}


if [ $MINIFY == "TRUE" ]; then
    echo "Minifying SVG figures."
    SOURCE_DIR=svg/minified
    for FIGURE in ${FIGURES[@]}
    do
        echo "Figure: " $FIGURE
        minify \
            --type svg --svg-precision 3 \
            --output ${FIGURE/svg/svg\/minified} \
            $FIGURE
    done
else
    SOURCE_DIR=svg
fi

echo "Exporting figures into PDF"
for FIGURE in ${FIGURES[@]}
do
    echo "Figure: " $FIGURE
    inkscape \
        --export-type=pdf \
        -o ${FIGURE//svg/pdf} \
        $FIGURE \
        2> /dev/null
done

pdfunite \
    ${MAIN_FIGURES[@]//svg/pdf} \
    MainFigures.${CURRENT_DATE}.pdf
pdfunite \
    ${SUPP_FIGURES[@]//svg/pdf} \
    SupplementaryFigures.${CURRENT_DATE}.pdf
pdfunite \
    MainFigures.${CURRENT_DATE}.pdf \
    SupplementaryFigures.${CURRENT_DATE}.pdf \
    AllFigures.${CURRENT_DATE}.pdf

echo "Producing trimmed, unlabeled figures"
for FIGURE in ${FIGURES[@]}
do
    echo "Figure: " $FIGURE
    NUM=`echo $FIGURE | tr -dc '0-9'`
    if [[ "$FIGURE" == *"FigureS"* ]]; then
        sed \
            "s/Supplementary Figure $NUM//g" $FIGURE \
            > ${FIGURE/.svg/.trimmed.svg}
    else
        sed \
            "s/Figure $NUM//g" $FIGURE \
            > ${FIGURE/.svg/.trimmed.svg}
    fi
    OUTPUT=${FIGURE/.svg/.trimmed.pdf}
    inkscape \
        --export-area-drawing \
        --export-margin=5 \
        --export-type=pdf \
        -o ${OUTPUT/svg/pdf} \
        ${FIGURE/.svg/.trimmed.svg} \
        2> /dev/null
    OUTPUT=${FIGURE/.svg/.trimmed.png}
    inkscape \
        --export-area-drawing \
        --export-margin=5 \
        --export-background=white \
        --export-dpi=$DPI \
        --export-type=png \
        -o ${OUTPUT/svg/png} \
        ${FIGURE/.svg/.trimmed.svg} \
        2> /dev/null
done

PDFS=${MAIN_FIGURES[@]//svg/pdf}
pdfunite \
    ${PDFS[@]/.pdf/.trimmed.pdf} \
    MainFigures.${CURRENT_DATE}.trimmed.pdf
PDFS=${SUPP_FIGURES[@]//svg/pdf}
pdfunite \
    ${PDFS[@]/.pdf/.trimmed.pdf} \
    SupplementaryFigures.${CURRENT_DATE}.trimmed.pdf
pdfunite \
    MainFigures.${CURRENT_DATE}.trimmed.pdf \
    SupplementaryFigures.${CURRENT_DATE}.trimmed.pdf \
    AllFigures.${CURRENT_DATE}.trimmed.pdf


if [ $CLEANUP_TEMP == "TRUE" ]; then
    rm ${FIGURES[@]/.svg/.trimmed.svg}

    PDFS=${FIGURES[@]/svg/pdf}
    rm ${PDFS[@]/.pdf/.trimmed.pdf}
fi
