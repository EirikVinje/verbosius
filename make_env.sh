#!/bin/bash

echo "Setting up environment..."

cd ~
mkdir -p data && cd data
mkdir -p verbosius && cd verbosius
mkdir -p hpsearch_env && cd hpsearch_env
mkdir -p amazon && cd amazon

mkdir -p chunking && rm -rf chunking/*
mkdir -p preprocess && rm -rf preprocess/*
mkdir -p trainingdata && rm -rf trainingdata/*
mkdir -p models && rm -rf models/*