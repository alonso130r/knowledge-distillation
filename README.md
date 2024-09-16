# knowledge-distillation
**Built with Llama**
Research platform for testing whether prompting strategies combined with knowledge distillation can increase the efficiency and accessibility of locally run LLMs.

For more details see the [research paper]().

# Setup

NOTE - running the setup is difficult, due to the EXLlamaV2 library. Some of its details are finicky and less than straightforward to use. 

## Dependencies:
Python Version: 3.11.9
Torch Version: 2.4.0


EXLlamaV2(link): V0.1.8
Note - this is only for systems x86 linux systems, with C++311. To run on other systems, see the EXLlamaV2 Github Releases and select the appropriate version of the 0.1.8 release. 
```
pip install https://github.com/turboderp/exllamav2/releases/download/v0.1.8/exllamav2-0.1.8+cu121.torch2.4.0-cp311-cp311-linux_x86_64.whl
```

## Quantization

Llama 3.1 405B Instruct was used for this study. A 6bit (average) quantized version was used, to reduce the cost of inference. Unfortunately, no API (known to the researchers) can be used as the study trains the student model directly on the logits outputted from the teacher model (using white-box knowledge distillation), and no known API outputs Llama 3.1 405B logits. 

## Training


## Evaluation
The student model was evaluated using a jupyter notebook. The results from the evaluation are in /logs

When no answer was detected, -1 was put as the numerical value, as there are no instances of -1 being an answer in GSM8K test set.

To extract numerical answers from the LLM responses, the following logic was used (implemented in the jupyter notebook using Regex), ordered by decreasing precedence:
1. Looks for a \boxed{} label in the answer, uses the first one found.
2. Looks at the first line of the answer, checking if there are any numbers on that line. If there is more than one, it only accepts if they're the same number.
3. Looks for "the [final/correct] answer is" phrase, and takes the first number following that phrase.
4. Looks for all the numbers in the answer, chooses the one closest to the end of the answer. 

If none of these automatic extraction rules works, or if there is an issue, it is passed for manual extraction. This occured ~120 times out of GSM8K test (1319).
