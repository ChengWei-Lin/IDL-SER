
# run all runs: specify date then run that date only
# for wandb_dir in $(find . -type d -name "run-20250217*"); do
#     echo "wandb_dir: ${wandb_dir}"
#     config_file="${wandb_dir}/files/config.yaml"
#     python3 project/test.py --config_file=$config_file
# done
# find all runs
# find . -type d -newermt "2025-02-12 12:09" | wc -l

# run particular run id
run_id="uy3pgzms"
wandb_dir=$(find . -type d -name "*${run_id}*" | head -n 1)
config_file="${wandb_dir}/files/config.yaml"

echo "wandb_dir: ${wandb_dir}"
echo "config_file: ${config_file}"

python3 project/test.py --config_file=$config_file