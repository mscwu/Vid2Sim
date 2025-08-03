# Generate depth prior for the scene
seq_path=$1

cd tools/
python generate_depth.py $seq_path