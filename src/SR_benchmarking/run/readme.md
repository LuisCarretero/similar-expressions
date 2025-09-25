# Clean install

## Julia setup

- Install Julia 11.x (e.g., via `juliaup`).
- Activate the project at `/cephfs/home/lc865/workspace/similar-expressions/Project.toml`.
- Dev SymbolicRegression:
  ```julia
  ] dev /cephfs/home/lc865/workspace/similar-expressions/SymbolicRegression
  ```
- Run the script at `/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/run/run_inference.jl` to verify that the neural backend works (requires GPU).
  - A successful run will output the usual SR.jl logs, e.g.,
    ```
    [ Info: Results saved to: outputs/20250919_153117_IDm9h7/hall_of_fame.csv
    ```

## Create Python environment

- Create and activate a conda environment:
  ```bash
  conda create -n SR-inference python=3.12
  conda activate SR-inference
  ```
- Install Python dependencies:
  ```bash
  pip install -r /cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/requirements.txt
  ```

## PySR setup

- Update `juliapkg.json` with the SR.jl and project path:

  ```json
  {
      "julia": "1.11",
      "packages": {
          "SymbolicRegression": {
              "uuid": "8254be44-1295-4e6a-a16d-46603ac705cb",
              "dev": true,
              "path": "/cephfs/home/lc865/workspace/similar-expressions/SymbolicRegression"
          },
          "Serialization": {
              "uuid": "9e88b42a-f829-5b0c-bbe9-9e923198166b",
              "version": "1"
          },
          "Revise": {
              "uuid": "295af30f-e4ad-537b-8983-00126c2a3abe"
          },
          "DynamicExpressions": {
              "uuid": "a40a106e-89c9-4ca8-8020-a735e8728b6b"
          }
      },
      "project": "/cephfs/home/lc865/workspace/similar-expressions"
  }
  ```

- Install PySR:
  ```bash
  cd PySR && pip install -e .
  ```

---

# Running the benchmark

1. Update `log_dir` and `model_path` in both:
   - `SR_benchmarking/run/config_neural.yaml`
   - `SR_benchmarking/run/config_vanilla.yaml`

   Use absolute paths. The model is in `/cephfs/home/lc865/workspace/similar-expressions/onnx-models/model-e51hcsb9.onnx`.

2. Edit SLURM settings in:
   ```
   /cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/run/slurm_distributed_run.sh
   ```
   > Note: `TOTAL_NODES` must equal the SLURM array size to correctly subsample the dataset on each node.

3. Submit the job:
   ```bash
   sbatch /cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/run/slurm_distributed_run.sh
   
