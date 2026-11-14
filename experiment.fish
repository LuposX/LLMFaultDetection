 for n in 1 2
       python3 src/run_random_baseline.py --faults 385 --n-patterns $n --repeats 5 \
           --out results/pattern_budget/random_n$n.jsonl 2>&1 | tee results/pattern_budget/random_n$n.log
   end
   for n in 1 2
       python3 src/run_hotflip_baseline.py --faults 385 --hf-starts $n --start-sampling random \
           --repeats 5 --out results/pattern_budget/hotflip_n$n.jsonl 2>&1 | tee
 results/pattern_budget/hotflip_n$n.log
   end
