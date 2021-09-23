rsync -arvz --progress --exclude="**/checkpoints" -e 'ssh -i id_rsa_keyless' dberger@siku.ace-net.ca:/home/dberger/projects/def-jlevman/dberger/ABIDE-eig/lightning_logs .
