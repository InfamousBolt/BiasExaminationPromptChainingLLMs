# Examining Bias in LLM Prompt Chaining

![image](https://github.com/user-attachments/assets/fe2a0e86-292c-4382-9877-f2583e802efe)

> A comprehensive investigation of how prompt chaining techniques affect the propagation of bias in Large Language Model outputs, using the StereoSet dataset as a benchmark.

## Project Overview

This project explores how prompt chaining—a technique where each step's output feeds into the next prompt—influences bias propagation in Large Language Models (LLMs). Due to their generative nature, LLMs may produce outputs containing unintended biases. We investigate whether sequential prompting techniques can mitigate or potentially amplify stereotypical associations in model responses.

Using the StereoSet dataset as a benchmark, we compare bias levels between direct (single-step) prompting and chained (multi-step) prompting across different stereotype categories including gender, race, religion, and profession.

## Research Questions

This project addresses three key research questions:

1. How does multi-step prompt chaining affect the propagation and amplification of social biases in LLM-generated outputs?
2. Which steps of a prompt chain are more likely to amplify stereotypes (if any)? Does this suggest an optimal chain length to minimize bias?
3. Do certain classes of stereotypes more commonly get amplified when using prompt chaining?

## Methodology & Implementation

### Dataset

We utilize the StereoSet dataset, which provides:
- Context statements that may evoke stereotypical associations
- Three completion options for each context: stereotypical, anti-stereotypical, and unrelated
- Annotations across four main bias categories: gender, race, religion, and profession

### Prompting Approaches

1. **Direct Prompting (Baseline)**:
   - Single-step prompting using the format: `{context} In a neutral manner, what happened next?`
   - Used as control condition to establish baseline bias levels

2. **Prompt Chaining (Experimental)**:
   - Three-step sequential chain:
     - Step 1: Baseline continuation of the context in a neutral manner
     - Step 2: Bias triggering frame to identify implicit assumptions
     - Step 3: Free generation avoiding identified stereotypes
   - Variations with different chain lengths (2-5 steps) to identify optimal configurations

### Bias Quantification

We developed a novel scoring system based on similarity measurements:
- **Stereotype Score**: Jaccard similarity between model response and stereotypical completions
- **Anti-stereotype Score**: Jaccard similarity between model response and anti-stereotypical completions
- **Bias Score**: Computed as (Stereotype Score - Anti-stereotype Score)
  - Positive values indicate bias toward stereotypes
  - Negative values indicate bias toward anti-stereotypes
  - Values near zero suggest neutrality

### Implementation Details

The experimental pipeline includes:
- Stratified sampling across stereotype categories
- Parallel evaluation of direct and chained prompting
- Chain progression analysis tracking bias through sequential steps
- Comparative analysis across stereotype categories
- Qualitative analysis of representative examples

All experiments were implemented using PyTorch with CUDA acceleration on GPU hardware.

## Key Findings

Our research led to several important insights:

1. **Chain Length Effects**:
   - Shorter chains (2-3 steps) generally reduce bias compared to direct prompting
   - Longer chains (4+ steps) tend to reintroduce or amplify bias
   - Non-linear relationship between chain depth and bias mitigation

2. **Category-Specific Effects**:
   - Gender and profession stereotypes were more susceptible to bias propagation
   - Race and religion categories showed more consistent bias reduction through chaining

3. **Model Robustness**:
   - Modern LLMs (we used LLaMA 3.1 8B-Instruct) exhibit generally low bias scores
   - Even with intentional bias triggers, models tend to produce relatively balanced outputs

4. **Bias Evolution**:
   - Step 2 (bias awareness) often showed the lowest bias scores
   - Final outputs sometimes reintroduced stereotypical associations despite explicit instructions

## Visualization Results

The repository includes several key visualizations:
- Bias progression through chain steps
- Comparison of direct vs. chained prompting across categories
- Distribution of bias scores for different chain architectures
- Detailed analysis of stereotype vs. anti-stereotype scores

![image](https://github.com/user-attachments/assets/e8629934-e070-43fd-9206-9e29e74719db)

## Usage & Reproduction

### Dependencies

```
torch
transformers
datasets
pandas
numpy
matplotlib
seaborn
tqdm
```

### Running the Experiments

```python
# Set the model you want to test
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Run the experiment with a specific number of examples
direct_results, chained_results, comparison, by_category = run_bias_experiment(
    model_name,
    num_examples=100
)

# Analyze chain progression
progression_df, step_averages, type_step_averages = analyze_chain_progression(chained_results)

# Create qualitative analysis
interesting_examples, report_text = create_qualitative_analysis(direct_results, chained_results)

# Experiment with different chain architectures
direct_results, architecture_results, architecture_scores, type_scores = experiment_with_chain_architectures(
    model_name,
    num_examples=50
)
```

## Practical Applications

Our findings have important implications for AI system design:

1. **Optimal Chain Architecture**: Using 2-3 step chains with explicit bias awareness prompts can reduce stereotypical associations in model outputs.

2. **Domain-Specific Adjustments**: Extra caution is warranted when using prompt chaining for topics related to gender and profession, where bias amplification is more likely.

3. **User Interaction Design**: Systems can incorporate explicit stereotype-avoidance instructions in multi-turn dialogue systems without excessive chain length.

## Conclusion & Future Work

This research demonstrates that prompt chaining can be an effective technique for bias mitigation when properly configured, but requires careful design to avoid unintended amplification effects. Future work may explore:

- Dynamic chain length adjustment based on context sensitivity
- Hybrid strategies combining bias detection and chained prompting
- Cross-dataset and cross-lingual evaluations to assess broader applicability
- Integration with reinforcement learning from human feedback for adaptive chain design

## Contributors

This project was developed by:
- Keshav Agarwal (keshav6@illinois.edu)
- Yash Telang (ytelang2@illinois.edu)
- Avani Puranik (avanip2@illinois.edu)
- Arthur Wang (arthurw3@illinois.edu)

## References

1. Guo, Y., et al. (2024). Bias in large language models: Origin, evaluation, and mitigation. arXiv preprint arXiv:2411.10915v1.
2. He, K., Long, Y., and Roy, K. (2024). Prompt-based bias calibration for better zero/few-shot learning of language models.
3. Kamruzzaman, M., and Kim, G. L. (2024). Prompting techniques for reducing social bias in LLMs through system 1 and system 2 cognitive processes. arXiv preprint arXiv:2404.17218.
4. Nadeem, M., Bethke, A., and Reddy, S. (2021). StereoSet: Measuring stereotypical bias in pretrained language models. In Proceedings of the ACL.
5. Raza, S., Raval, A., and Chatrath, V. (2024). MBias: Mitigating bias in large language models while retaining context. arXiv preprint arXiv:2405.11290.
6. Wu, T., Terry, M., and Cai, C. J. (2022). AI Chains: Transparent and controllable human-AI interaction by chaining large language model prompts. In Proceedings of the 2022 CHI Conference.
