set -e

--dataset_dirs / --checkpoint_dir / --test_dirs are all resolved against
--repo_root, which probe_common.py defaults to the mats12/ submodule
directory (two levels above probe_common.py), not this repo's root. The
datasets_deepseek_gullibility_v0_3.deduped.3992/ dataset lives at the
SeeGULL repo root, one level above mats12/, hence the leading "../".

python mats12/src/train_reading_probe.py --dataset_dirs ../datasets_deepseek_gullibility_v0_3.deduped.3992/train/ --output_dir "probe_checkpoints.4k" --run_name reading_probe

python mats12/src/train_control_probe.py --dataset_dirs ../datasets_deepseek_gullibility_v0_3.deduped.3992/train/ --output_dir "probe_checkpoints.4k" --run_name control_probe

python mats12/src/test_reading_probe.py --checkpoint_dir "probe_checkpoints.4k/reading_probe" --test_dirs ../datasets_deepseek_gullibility_v0_3.deduped.3992/holdout/

python mats12/src/test_control_probe.py --checkpoint_dir "probe_checkpoints.4k/control_probe" --test_dirs ../datasets_deepseek_gullibility_v0_3.deduped.3992/holdout
