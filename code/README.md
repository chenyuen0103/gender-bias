# Gender Bias Project
This repo contains the code and data for the paper:
[**Understanding Stereotypes in Language Models: Towards Robust Measurement and Zero-Shot Debiasing**](placeholder for arxiv link)

*Justus Mattern, Yuen Chen, Mrinmaya Sachan, Rada Mihalcea, Bernhard Schölkopf, and Zhijing Jin*

### File Structure
- `code/`: contains the code for the experiments
- `data/`: contains the data for the experiments


### How to Run
1. Setup environment:
   - run `pip install -r requirements.txt`
2. Run experiments on explicit gender bias (first task prompt in Table 1):
   - run `python code/pipleline_genderquestion.py` for prompt without conversation.
   - run `python code/pipeline_genderquestion_conv.py` for prompt with conversation.
3. Run experiment on implicit gender bias (second to fourth task prompts in Table 1):
   - run `python code/pipeline.py` for prompt without conversation.
   - run `python code/pipeline_conversation.py` for prompt with conversation.

4. The results will be saved in `data/` folder.

