for i in 3 5 25 
do
    for opt_mod in 'SPYNET' 'RAFT'
    do
        export rolling_window=$i
        export optical_flow_module=$opt_mod
        export exp_name="${opt_mod}_bvsr_${rolling_window}"

        echo "Running experiment: ${exp_name}"
        
        python src/test_basic_vsr.py -c tests/config_bvsr_test_raft_tpl.cfg -v
    done
done

