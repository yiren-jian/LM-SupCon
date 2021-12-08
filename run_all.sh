for seed in 13 21 42 87 100
do
    for bs in 40
    do
        for lr in 1e-5
        do
            for supcon_lr in 1e-5
            do
                TAG=LM-BFF \
                TYPE=prompt-demo \
                TASK=sst-5 \
                BS=$bs \
                LR=$lr \
                SupCon_LR=$supcon_lr \
                SEED=$seed \
                MODEL=roberta-base \
                bash run_experiment.sh
            done
        done
    done
done

rm -rf result/

for seed in 13 21 42 87 100
do
    for bs in 16
    do
        for lr in 1e-5
        do
            for supcon_lr in 1e-5
            do
                TAG=LM-BFF \
                TYPE=prompt-demo \
                TASK=CoLA \
                BS=$bs \
                LR=$lr \
                SupCon_LR=$supcon_lr \
                SEED=$seed \
                MODEL=roberta-base \
                bash run_experiment.sh
            done
        done
    done
done

rm -rf result/

for seed in 13 21 42 87 100
do
    for bs in 32
    do
        for lr in 1e-5
        do
            for supcon_lr in 1e-5
            do
                TAG=LM-BFF \
                TYPE=prompt-demo \
                TASK=trec \
                BS=$bs \
                LR=$lr \
                SupCon_LR=$supcon_lr \
                SEED=$seed \
                MODEL=roberta-base \
                bash run_experiment.sh
            done
        done
    done
done

rm -rf result/

for seed in 13 21 42 87 100
do
    for bs in 24
    do
        for lr in 1e-5
        do
            for supcon_lr in 1e-5
            do
                TAG=LM-BFF \
                TYPE=prompt-demo \
                TASK=MNLI \
                BS=$bs \
                LR=$lr \
                SupCon_LR=$supcon_lr \
                SEED=$seed \
                MODEL=roberta-base \
                bash run_experiment.sh
            done
        done
    done
done

rm -rf result/

for seed in 13 21 42 87 100
do
    for bs in 32
    do
        for lr in 1e-5
        do
            for supcon_lr in 1e-5
            do
                TAG=LM-BFF \
                TYPE=prompt-demo \
                TASK=SNLI \
                BS=$bs \
                LR=$lr \
                SupCon_LR=$supcon_lr \
                SEED=$seed \
                MODEL=roberta-base \
                bash run_experiment.sh
            done
        done
    done
done

rm -rf result/

for seed in 13 21 42 87 100
do
    for bs in 16
    do
        for lr in 1e-5
        do
            for supcon_lr in 1e-5
            do
                TAG=LM-BFF \
                TYPE=prompt-demo \
                TASK=QNLI \
                BS=$bs \
                LR=$lr \
                SupCon_LR=$supcon_lr \
                SEED=$seed \
                MODEL=roberta-base \
                bash run_experiment.sh
            done
        done
    done
done

rm -rf result/

for seed in 13 21 42 87 100
do
    for bs in 32
    do
        for lr in 1e-5
        do
            for supcon_lr in 1e-5
            do
                TAG=LM-BFF \
                TYPE=prompt-demo \
                TASK=QQP \
                BS=$bs \
                LR=$lr \
                SupCon_LR=$supcon_lr \
                SEED=$seed \
                MODEL=roberta-base \
                bash run_experiment.sh
            done
        done
    done
done

rm -rf result/

for seed in 13 21 42 87 100
do
    for bs in 32
    do
        for lr in 1e-5
        do
            for supcon_lr in 1e-6
            do
                TAG=LM-BFF \
                TYPE=prompt-demo \
                TASK=RTE \
                BS=$bs \
                LR=$lr \
                SupCon_LR=$supcon_lr \
                SEED=$seed \
                MODEL=roberta-base \
                bash run_experiment.sh
            done
        done
    done
done

rm -rf result/

for seed in 13 21 42 87 100
do
    for bs in 16
    do
        for lr in 1e-5
        do
            for supcon_lr in 1e-5
            do
                TAG=LM-BFF \
                TYPE=prompt-demo \
                TASK=MRPC \
                BS=$bs \
                LR=$lr \
                SupCon_LR=$supcon_lr \
                SEED=$seed \
                MODEL=roberta-base \
                bash run_experiment.sh
            done
        done
    done
done

rm -rf result/

for seed in 13 21 42 87 100
do
    for bs in 16
    do
        for lr in 1e-5
        do
            for supcon_lr in 1e-6
            do
                TAG=LM-BFF \
                TYPE=prompt-demo \
                TASK=mr \
                BS=$bs \
                LR=$lr \
                SupCon_LR=$supcon_lr \
                SEED=$seed \
                MODEL=roberta-base \
                bash run_experiment.sh
            done
        done
    done
done

rm -rf result/

for seed in 13 21 42 87 100
do
    for bs in 16
    do
        for lr in 1e-5
        do
            for supcon_lr in 1e-5
            do
                TAG=LM-BFF \
                TYPE=prompt-demo \
                TASK=mpqa \
                BS=$bs \
                LR=$lr \
                SupCon_LR=$supcon_lr \
                SEED=$seed \
                MODEL=roberta-base \
                bash run_experiment.sh
            done
        done
    done
done

rm -rf result/

for seed in 13 21 42 87 100
do
    for bs in 32
    do
        for lr in 1e-5
        do
            for supcon_lr in 1e-5
            do
                TAG=LM-BFF \
                TYPE=prompt-demo \
                TASK=cr \
                BS=$bs \
                LR=$lr \
                SupCon_LR=$supcon_lr \
                SEED=$seed \
                MODEL=roberta-base \
                bash run_experiment.sh
            done
        done
    done
done

rm -rf result/

for seed in 13 21 42 87 100
do
    for bs in 16
    do
        for lr in 1e-5
        do
            for supcon_lr in 1e-6
            do
                TAG=LM-BFF \
                TYPE=prompt-demo \
                TASK=SST-2 \
                BS=$bs \
                LR=$lr \
                SupCon_LR=$supcon_lr \
                SEED=$seed \
                MODEL=roberta-base \
                bash run_experiment.sh
            done
        done
    done
done

rm -rf result/

for seed in 13 21 42 87 100
do
    for bs in 16
    do
        for lr in 1e-5
        do
            for supcon_lr in 1e-5
            do
                TAG=LM-BFF \
                TYPE=prompt-demo \
                TASK=subj \
                BS=$bs \
                LR=$lr \
                SupCon_LR=$supcon_lr \
                SEED=$seed \
                MODEL=roberta-base \
                bash run_experiment.sh
            done
        done
    done
done

rm -rf result/
