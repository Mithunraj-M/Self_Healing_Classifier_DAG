# Self Healing Classifier using DAG
### This DAG uses Langgraph, DistillBERT Transformer to classifiy emotions based on textual input. This workflow involves self healing classification when confidence score is low using fallback model and user clarification.
## How To Setup?
### Step1: 
- Clone the repository git clone https://github.com/Mithunraj-M/Self_Healing_Classifier_DAG.git

### Step2: 
- pip install -r requirements.txt

### Step3:
- In CLI run python script: python langgraph_app.py

## Implementation
## Fine tuned Transformer Model
- fine tuned DistillBERT using LoRA 
- Dataset used: dair-ai/emotion (six classes['joy','sadness','anger','surprise','love','fear'])
- Validation Accuracy 91.75%
- Training Accuracy 92.88%
### Performance Metrics

![Training Log](imgs/epochs.png)

### Final Train Metrics:
- train_loss: 0.1853
- train_accuracy: 0.9288
- train_precision: 0.9292
- train_recall: 0.9288
- train_f1: 0.9289

### Final Validation Metrics:
- eval_loss: 0.2283
- eval_accuracy: 0.9175
- eval_precision: 0.9187
- eval_recall: 0.9175
- eval_f1: 0.9175

![Metrics plots](imgs/output.png)

### confusion matrix
![Confusion Matrix](imgs/cm.png)

### Per class confidence score
![Per class confidence score](imgs/pcp.png)

### Mean Confidence Score
![Mean Confidence Score](imgs/mcp.png)

### LangGraph Workflow
![DAG](imgs/dag1.png)

## Inference Node

- Uses the saved transformer model to predict the emotion label
- Input: String 
       * Ex:"i can go from feeling so hopeless to so damned hopeful just from being around someone who cares and is awake" *
- Outputs the emotion prediciton and its confidence score

![Inference Output](imgs/Inference_output.png)
## Confidence Check Node

- Uses the Confidence Score computed by the inference node to check if the confidence score is above threshold
- If confidence Score is above threshold status is set to 'accept'
- else status is set to "fallback"moving to the Fallback node

![Confidence Check Node Output](imgs/Confidenc_output.png)

## FallBack Node

- Uses zero-shot classifier model : facebook/bart-large-mnli
- outputs emotion prediction and confidence score
- The confidence score is again checked against threshold and in case it is not accepted , it calls the clarification node

![Fallback Node Output](imgs/Fallback_output.png)

## Clarification Node

- Uses LLM Chains for asking clarification questions and augments to input text
- Augmented clarification text is passed on back to the fallback node for final prediction

![Clarification Node Output](imgs/clarification_output.png)

## Visualizer 

- provides plots of 'Prediction Confidence Distribution' , 'Mean Confidence by Label' , 'Prediction Outcome Frequency'
- visualizer makes use of the data monitored and logged in the logs.

## Logs

- Logs at each node is monitored and logged into the logs folder present in the repository
- https://github.com/Mithunraj-M/Self_Healing_Classifier_DAG/tree/master/logs

# Working Examples
## Example1: 
- Input: I went to mall to get a gift 
- Expected Prediction : joy
- Output prediction: joy
- Confidence Score: 0.78
- Fallback not triggered

![example1](imgs/ex_1.png)

## Example2:
- Input: I finished all my work on time
- Expected Prediction : joy
- Inference output: saddness
- Inference Confidence Score: 0.39
- Fallback triggered
- Fallback Output: surprise
- Fallback Confidence Score: 0.35
- Clarification triggered
- Clarification Output: joy
- Final Confidence Score: 0.83
![example2](imgs/ex_2.png)

# Video Demo https://drive.google.com/file/d/1hr1K-zL12UTmrGX05XzOFBewn3d9T0jq/view?usp=sharing
## Visualization using Log Data

![figure1](imgs/Figure_1.png) ![figure2](imgs/Figure_2.png) ![figure3](imgs/Figure_3.png)