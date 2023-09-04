#!/bin/bash

cd ~
mkdir -p data && cd data
mkdir -p verbosius && cd verbosius
mkdir -p imdb && cd imdb
mkdir -p testing && cd testing

mkdir -p chunking && rm -rf chunking/*
mkdir -p preprocess && rm -rf preprocess/*
mkdir -p trainingdata && rm -rf trainingdata/*
mkdir -p models && rm -rf models/*


