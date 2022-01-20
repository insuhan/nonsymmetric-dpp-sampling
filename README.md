# Scalable Sampling for Nonsymmetric Determinantl Point Processes

Python Implmentation for Scalable Sampling for Nonsymmetric Determinantl Point Processes


# Organization
- The code files are organized for (1) sampling nonymmetric DPPs (NDPPs) and (2) learning with orthogonality constraints.
- The code is based on https://github.com/cgartrel/nonsymmetric-DPP-learning/ (Scalable Learning and MAP Inference for Nonsymmetric Determinantal Point Processes, ICLR 2021)

# Usage

## Experiments for learning orthonal NDPPs:

- First, download the datasets 
    ```
    bash download.sh
    ```

- To run the ONDPP learn with UK Retail dataset,
    ```
    cd ./learning
    bash script_ondpp.sh
    ```

## Experiments for scalable sampling from NDPPs:

- To run synthetic dataset, 
    ```
    cd ./sampling
    python run_synthetic.py
    ```

- This will run the Cholesky-based sampling and tree-based rejection sampling
- Parameters in ``run_synthetic.py'' (e.g., ground set size) can be changed 
- Models learned from real-world datasets also can be used for sampling (parameters are saved in ../learning/saved_models/)
