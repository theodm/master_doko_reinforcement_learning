source "$HOME/.rye/env"
source "$HOME/.cargo/env"

cd rs-doko-py-bridge

rye sync

source .venv/bin/activate

maturin develop --release

cd ..

mkdir -p output

cd output

python ../rs-doko-py-bridge/python/run_mcts_vs_prev.py

