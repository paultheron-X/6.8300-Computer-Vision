

for i in 3 5 25 
do
    for j in 0 1 
    do
        export rolling_window=$i
        export reset_spynet=$j
        export exp_name="bvsr_${rolling_window}_reset${reset_spynet}"

        echo "Running experiment: ${exp_name}"
        
        python src/test_basic_vsr.py -c tests/config_bvsr_test_tpl.cfg -v
    done
done

