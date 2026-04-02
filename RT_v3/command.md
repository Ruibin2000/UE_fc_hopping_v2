tmux new -s caprun

conda activate sionna-gpu
mkdir -p logs
python -u RT_v3/Capacity_map_TX2.py |& tee -a logs/caprun_g500_s2.log
    # 断开：Ctrl+b d

tmux ls
tmux attach -t caprun

tail -f logs/caprun.log