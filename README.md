# comparison experiments
This is the code for comparison experiments with the following three methods.The design and implementation follows original paper as much as possible.

# Method


|   Mothods   | Paper                                                                                                                        | 
| :-------: |:-----------------------------------------------------------------------------------------------------------------------------|
| Mul-SAN  | [2023][MulOER-SAN: 2-layer multi-objective framework for exercise recommendation with self-attention networks](https://pan.baidu.com/s/13wnlY-dT7ivmrXavzriwYg?pwd=d2qu) | 
|   KG4Ex    | [2023][KG4Ex: An explainable knowledge graph-based approach for exercise recommendation](https://pan.baidu.com/s/1euY_3afXoST9REev1GCgNw?pwd=z1lx)                          | 
| FC-CDF  | [ 2023][Predicting examinee performance based on a fuzzy cloud cognitive diagnosis framework in e-learning environment](https://pan.baidu.com/s/1RWnwJ1RjQAysC0KzHdgS-A?pwd=sa1k)      | 

# Project Structure

- `code`: Implementation of Mul-SAN
- `data`
  - `data`: Processed data
  - `MOOPer`: Dataset used
  - `save_data`: Results saved
- `experiment of KG4Ex`
  - `Feature_extraction`: Student feature extraction
    - `code`: Code for student feature extraction
    - `data`
      - `data`: Process data used in feature extraction
      - `MOOPer`: Dataset used
  - `graph&recommend`: Constructing graph and recommending Exercises
    - `code`: Code for constructing graph and recommending Exercises
    - `data`
      - `data`
        - `add_relation`: Process data used in constructing graph
        - `entity`: All entities of the graph
        - `kg`: All triplets of the graph
        - `old_relation`: Process data in building relation
        - `relation`: All relations of the graph
      - `MOOPer`: Dataset used
      - `recent_6_4`: Results saved
- `fc_cdf`: Implementation of FC_CDF
- `result`: Comparison result between FC-CDF and other CDMs
