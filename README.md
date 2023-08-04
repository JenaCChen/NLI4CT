
# Brandeis CLMS Capstone 2023
## Task 7: Multi-evidence Natural Language Inference for Clinical Trial Data (NLI4CT) at SemEval 2023

This project aims to train and implement two machine learning models, leveraging BERT-based Language Models (LLMs), to address the Textual Entailment and Evidence Retrieval sub-tasks in [NLI4CT](https://sites.google.com/view/nli4ct/home?authuser=0). Additionally, we provide a user-friendly interface using Flask, allowing users to make their own inferences using the trained models.

## Dataset

The NLI4CT dataset can be accessed on [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/8937#learn_the_details-dataset).

## Environment Requirements

To set up the required environment, install the necessary dependencies using the provided `requirements.txt` file:

<pre>
pip install -r requirements.txt
</pre>

## Training the Model

The model training involves two steps. Firstly, we train the Textual Entailment model, followed by the Evidence Retrieval model. Both models are saved upon successful completion of the training process.

The model training offers flexibility with adjustable parameters. The following parameters can be adjusted:

- Learning Rate
- Batch Size
- Number of Epochs
- BERT Models
- Feature Combination
- Include SpaCy NER Tagger

To train the models, use the following command with the required parameters:
<pre>
python train.py --model model_name --batch_size batch_size_value --lr learning_rate_value --epoch num_epochs --mode combination_option --ner
</pre>

Please replace `model_name`, `batch_size_value`, `learning_rate_value`, `num_epochs`, and `combination_option` with the desired values for the respective parameters.

## User Interface

Our user interface leverages the pre-trained models to make inferences. If the models have not been trained previously, running the following command will automatically initiate the training process:
<pre>
python run.py
</pre>

Once the models are trained or loaded, you can interact with the interface by visiting http://127.0.0.1:5000/ in your web browser.
