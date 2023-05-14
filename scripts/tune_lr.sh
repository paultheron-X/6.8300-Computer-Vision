
for lr_f in 1e-5 5e-5 1e-4 5e-4
do
    for lr_b in 1e-5 5e-5 1e-4 5e-4
    do
        export lr_finetune=$lr_f
        export lr_base=$lr_b
        export exp_name="experiment_lrf_${lr_f}_lrb_${lr_b}"

        echo "Running experiment: ${exp_name}"
        
        python src/train_multistage_bvsr.py -c train/config_mstagebvsr_tpl.cfg -v
    done
done