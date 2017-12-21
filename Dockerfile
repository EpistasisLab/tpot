FROM python:2.7

LABEL vendor="DeepLearn Inc"

ENV WORKDIR=/var/src/tpot-1

COPY tpot $WORKDIR/tpot

