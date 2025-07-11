# In das Verzeichnis wechseln, in dem sich dieses Skript befindet.
cd "$(dirname "$0")"

cd ./rs-doko-py-bridge

# venv verlassen, falls aktiv (wir starten alles ohne venv)
deactivate
conda init
conda deactivate

maturin develop


