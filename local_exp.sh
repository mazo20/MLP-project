#USAGE: bash run_exp.sh <gen experiment file> <name of the experiment>
#EXAMPLE: bash run_exp.sh gen_best.py bestModel
mkdir -p results/$1
zip -r code.zip code
mv -f code.zip results/$1

python3 code/main.py --results_root results/$1
