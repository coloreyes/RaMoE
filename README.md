# RaMoE

RaMoE (Routing Mixture of Experts) is a deep learning project designed to implement an efficient mixture of experts model. It provides modules for data preprocessing, model training, experiment management, and result analysis.

## Project Structure

```
RaMoE/
├── datasets/
│ ├── ICIP
│ ├── INS
│ └── SMPD
├── model/
│ ├── Bert
│ ├── BLIP
│ └── VIT
├── src/
│ ├── preprocess/
│ ├── model/
│ └── RESULT/
└── README.md
```

## Environment Requirements

- Python >= 3.12.0
- Virtual environment recommended (venv or conda)

Install dependencies:
```bash
pip install -r requirements.txt
```
(If `requirements.txt` is missing, please install dependencies according to the code.)

## Quick Start

1. Clone the project
	```bash
	git clone https://github.com/coloreyes/RaMoE.git
	cd RaMoE
	```
2. Install dependencies
	```bash
	pip install -r requirements.txt
	```
3. Preprocess the data
	```bash
	python src/preprocess/preprocess_main.py --dataset ICIP --datasets_path ../../datasets --pretrained_model_path ../../model --retrieval_num 500 
	```
4. Run the main program
	```bash
	python src/main.py --dataset ICIP --datasets_path  ../datasets --mode train --moe_model RaMoE --num_experts 3 --retrieval_num 300 
	```

## Results and Outputs

All experiment results and model outputs are saved in the `src/RESULT/` directory.

---
