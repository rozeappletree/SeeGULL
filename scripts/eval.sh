# WARNING: Run block by block, overwrites eval folder in probe checkpoints folder
# TODO: Rename folder automatically, doing manually for now.


# python mats12/src/test_reading_probe.py --checkpoint_dir "probe_checkpoints.4k/reading_probe" --test_dirs /root/SeeGULL/mats12/datasets_defense_484
# python mats12/src/test_control_probe.py --checkpoint_dir "probe_checkpoints.4k/control_probe" --test_dirs /root/SeeGULL/mats12/datasets_defense_484


# python mats12/src/test_reading_probe.py --checkpoint_dir "probe_checkpoints.4k/reading_probe" --test_dirs /root/SeeGULL/mats12/datasets_hard_333
# python mats12/src/test_control_probe.py --checkpoint_dir "probe_checkpoints.4k/control_probe" --test_dirs /root/SeeGULL/mats12/datasets_hard_333


python mats12/src/test_reading_probe.py --checkpoint_dir "probe_checkpoints.4k/reading_probe" --test_dirs /root/SeeGULL/mats12/datasets_ardulous_66
python mats12/src/test_control_probe.py --checkpoint_dir "probe_checkpoints.4k/control_probe" --test_dirs /root/SeeGULL/mats12/datasets_ardulous_66

