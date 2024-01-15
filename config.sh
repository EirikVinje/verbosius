#!/bin/bash

dataset="$(python -c 'import config; print(config.dataset)')"
chunkdist_n="$(python -c 'import config; print(config.chunkdist_n)')"
chunk_size="$(python -c 'import config; print(config.chunk_size)')"
chunk_amount="$(python -c 'import config; print(config.chunk_amount)')"
