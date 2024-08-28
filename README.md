# comparison experiments
This is the code for comparison experiments with the following three methods.The design and implementation follows original paper as much as possible.

# Method


|   Mothods   | Paper                                                                                                                        | 
| :-------: |:-----------------------------------------------------------------------------------------------------------------------------|
| Mul-SAN  | [2023][MulOER-SAN: 2-layer multi-objective framework for exercise recommendation with self-attention networks](https://pan.baidu.com/s/13wnlY-dT7ivmrXavzriwYg?pwd=d2qu) | 
|   KG4Ex    | [2023][KG4Ex: An explainable knowledge graph-based approach for exercise recommendation](https://pan.baidu.com/s/1euY_3afXoST9REev1GCgNw?pwd=z1lx)                          | 
| FC-CDF  | [ 2023][Predicting examinee performance based on a fuzzy cloud cognitive diagnosis framework in e-learning environment](https://pan.baidu.com/s/1RWnwJ1RjQAysC0KzHdgS-A?pwd=sa1k)      | 

# File Details 
├─code: the code implementation of Mul-SAN
├─data
│  ├─data: process data
│  ├─MOOPer: the dataset used
│  └─save_data: the result saved
├─experiment of KG4Ex
│  ├─Feature_extraction: student feature extraction
│  │  ├─code: the code of student feature extraction
│  │  └─data
│  │      ├─data: process data used in feature extraction
│  │      └─MOOPer: the dataset used
│  └─graph&recommend: constructing graph and recommending exercises
│      ├─code: the code of constructing graph and recommending exercises
│      └─data
│          ├─data
│          │  ├─add_relation: process data used in constructing graph
│          │  ├─entity: all entities of the graph
│          │  ├─kg: all triplets of the graph
│          │  ├─old_relation: process data in building relation
│          │  └─relation: all relations of the graph
│          ├─MOOPer: the dataset used
│          └─recent_6_4: the result saved
├─fc_cdf: the code implementation of FC_CDF
│  ├─fccdf
│  │  ├─utils
│  │  │  └─__pycache__
│  │  └─__pycache__
│  ├─idea
│  │  └─inspectionProfiles
│  ├─math2015
│  │  ├─MOOPer
│  │  ├─real_data
│  │  ├─results
│  │  ├─TestData
│  │  └─testData2
│  ├─psy
│  │  ├─cat
│  │  │  └─__pycache__
│  │  ├─cdm
│  │  │  └─__pycache__
│  │  ├─ctt
│  │  ├─data
│  │  │  ├─data
│  │  │  └─__pycache__
│  │  ├─exceptions
│  │  │  └─__pycache__
│  │  ├─fa
│  │  │  └─__pycache__
│  │  ├─irt
│  │  │  └─__pycache__
│  │  ├─sem
│  │  │  └─__pycache__
│  │  └─settings
│  │      └─__pycache__
│  └─__pycache__
└─result: the comparison result between FC-CDF and other CDMs
