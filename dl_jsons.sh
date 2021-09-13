echo "Downloading Siku Optuna .json tables..."
rsync -arvz --progress --include "*.json" --include='*/' --exclude="*" -e 'ssh -i id_rsa_keyless' dberger@siku.ace-net.ca:/home/dberger/projects/def-jlevman/dberger/ABIDE-eig/htune_results htune_results/cc/ || echo "Failed to download from Siku" 

