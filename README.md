# Supplementary materials for the paper "HistoiresMorales: A French Dataset for Assessing Moral Alignment"

**Dataset is now also available on HuggingFace: https://huggingface.co/datasets/LabHC/histoires_morales.**

Aligning language models with human values is crucial, especially as they become more integrated into everyday life. 
While models are often adapted to user preferences, it is equally important to ensure they align with moral norms and behaviours in real-world social situations. 
Despite significant progress in languages like English and Chinese, French has seen little attention in this area, leaving a gap in understanding how LLMs handle moral reasoning in this language.
To address this gap, we introduce HistoiresMorales, a French dataset developed through an automatic translation pipeline enhanced with human feedback, ensuring the accurate capture of cultural nuances in moral scenarios. 
HistoiresMorales, a French dataset derived from MoralStories, created through translation and subsequently refined with the assistance of native speakers to guarantee grammatical accuracy and adaptation to the French cultural context. 
We also rely on annotations of the moral values within the dataset to ensure their alignment with French norms.
HistoiresMorales covers a wide range of social situations, including differences in tipping practices,
expressions of honesty in relationships, and responsibilities toward animals.
To foster future research, we also conduct preliminary experiments on the alignment of multilingual models on French and English data and the robustness of the alignment.
We find that while LLMs are generally aligned with human moral norms by default, they can be easily influenced with user-preference optimization for both moral and immoral data.


# Usage

1. `ppl.py` contains code for computing perplexity of moral and immoral text.
2. `declarative_prompt.py` contains code for experiments with declarative prompt.
3. `dpo.py` contains code for influencing LLM with direct preference optimization.
   
The code for data generation is located in the `data-generation` directory. 

- `translation.py`: Contains the code for data translation.
- `data-quality.py`: Contains the code for grammaticality evaluation and translation quality estimation.

Requirements for running the code are mentioned in ```requirements.txt```. Seeds and other parameters if any are listed in the scripts.

<details>
    <summary>Experiments on Model Moral Alignment</summary>

#### Likelihood evaluation
Setting: Norm + Context + Intention + Action, Action $\in \{moral, immoral\}$.
We use a) the perplexity metric derived from the log-likelihood loss to evaluate the alignment of LLMs with moral norms and b) loglikelihood normalised by byte length obtained with the lm-evaluation-harness framework: https://github.com/EleutherAI/lm-evaluation-harness.

#### Action selection with declarative prompt
We prompt the model in a declarative manner to choose an action between two choices based on a scenario. 
Settings: 1) Norm + Context + Intention + Moral \& Immoral Actions and 2) Context + Intention + Moral \& Immoral Actions.
We use the prompts mentioned in `declarative_prompt.py`. 
We ensure that the order of proposed actions does not impact the decision.

#### Influencing LLM with Direct Preference Optimization

We evaluate the robustness of LLM's moral alignment. 
Using Direct Preference Optimization (DPO): https://proceedings.neurips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf. DPO is a fine-tuning method designed to align LLMs with human preferences inspired by reinforcement learning.
We aim to influence the model to prefer either moral or immoral actions. 

</details>


