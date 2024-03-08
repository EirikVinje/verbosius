#!/bin/bash


dist=1
echo "clearing up chunkdist $dist..."

cd ~
mkdir -p data && cd data
mkdir -p verbosius && cd verbosius
mkdir -p amazon && cd amazon

mkdir -p chunking && rm -rf chunking/amazon_chunkdist_$dist
mkdir -p preprocess && rm -rf preprocess/amazon_chunkdist_$dist
rm -rf preprocess/amazon_chunkdist_$dist/*e.pkl
mkdir -p trainingdata && rm -rf trainingdata/amazon_chunkdist_$dist
mkdir -p models && rm -rf models/amazon_chunkdist_$dist

