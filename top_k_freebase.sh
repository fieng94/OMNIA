# Define the list of top-k values as an array
top_k=(1 2 3 4 5 10 15 20 50)
data='freebase'

# Loop through the array and run the Python program with each top-k value
for k in "${top_k[@]}"; do
    echo "Testing at top $k"
    python3 omnia_top_k.py --path "data/${data}/data_sample.csv" --output_dir "results/top_k/${data}" --setting triples --top_k $k
done
