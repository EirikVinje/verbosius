#!/bin/bash


dist=235987
echo "clearing up chunkdist $dist..."

cd ~
mkdir -p data && cd data
mkdir -p verbosius && cd verbosius
mkdir -p amazon && cd amazon

mkdir -p chunking && rm -rf chunking/amazon_chunkdist_$dist
mkdir -p preprocess && rm -rf preprocess/amazon_chunkdist_$dist
mkdir -p trainingdata && rm -rf trainingdata/amazon_chunkdist_$dist
mkdir -p models && rm -rf models/