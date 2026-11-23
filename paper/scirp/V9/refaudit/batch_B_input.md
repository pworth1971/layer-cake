## peters2018deep

```bibtex
@inproceedings{peters2018deep,
  title={Deep contextualized word representations},
  author={Peters, Matthew E. and Neumann, Mark and Iyyer, Mohit and Gardner, Matt and Clark, Christopher and Lee, Kenton and Zettlemoyer, Luke},
  booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)},
  address={New Orleans, Louisiana},
  pages={2227--2237},
  year={2018},
  publisher={Association for Computational Linguistics},
  doi={10.48550/arXiv.1802.05365},
  url={https://doi.org/10.48550/arXiv.1802.05365}
}
```

Citing contexts in the V6 manuscript:
- ...eration tools and AI more broadly over the last seven (7) years. Some of the early influential deep learning approaches to NLP which leveraged transformer architectures and attention mechanisms to compute contextualized (dynamic) text representation forms of natural language were ELMo (Embeddings from Language Models), which was introduced in 2018 \cite{peters2018deep} which used bidirectional LSTM layers to generate contextualized embeddings, and then of course BERT (Bidirectional Encoder Representations from Trans...

## radford2019language

```bibtex
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019},
  institution={OpenAI},
 url={https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf}}
```

Citing contexts in the V6 manuscript:
- ...amental linguistic theories (more on this below). Some of the most common methods of training include \textit{next token prediction}\footnote{Next token prediction based LLMs are called \textit{causal} language models due to the determinative nature of the underlying algorithm.} which is used for example by the GPT family of models built by OpenAI \cite{radford2019language}, and \textit{masking} which is used by the BERT \cite{devlin-etal-2019-bert} and RoBERTa \cite{liu2019roberta} based language model families models. ...
- ...ite{hinton2015distilling} to create a lighter, faster model while retaining most of BERT's performance. XLNet \cite{yang2019xlnet} builds on Transformer-XL \cite{dai2019transformer} and introduces permutation-based training to capture bidirectional context without masking, blending ideas from both encoder and autoregressive architectures and GPT-2 \cite{radford2019language} is an autoregressive, decoder-only model trained to predict the next token in a sequence, enabling strong generative capabilities. It serves as a con...
- ...formers library\footnote{https://huggingface.co/docs/transformers/en/index}. We support two primary architectures: the standard Sequence Classification model, which utilizes pretrained transformer models such as BERT \cite{devlin-etal-2019-bert}, RoBERTa \cite{liu2019roberta}, DistilBERT \cite{sanh2020distilbert}, XLNet \cite{yang2019xlnet}, GPT-2 \cite{radford2019language}, Llama \cite{touvron2023llama}, and DeepSeek \cite{guo2025deepseek}; and a transformer-based CNN architecture, integrating transformers' context-awar...
- ...all of the prior DL, neural network based architectures that came before it (e.g. RNNs with self-attention mechanisms or Convolutional Neural Networks or CNNs), underpinning many state-of-the-art results in machine translation, text classification \cite{devlin-etal-2019-bert}, natural language understanding, and (natural) language generation tools \cite{radford2019language} and applications. These models, and their associated platforms and operational infrastructures, have pushed the boundaries of language generation and...

## joachims1998text

```bibtex
@inproceedings{joachims1998text,
  title={Text categorization with Support Vector Machines: Learning with many relevant features},
  author={Joachims, Thorsten},
  booktitle={Proceedings of the European Conference on Machine Learning (ECML)},
  pages={137--142},
  year={1998},
  publisher={Springer},
  address={Berlin, Heidelberg}
}
```

Citing contexts in the V6 manuscript:
- ... these performance gains. Traditionally, the problem of text classification has been approached through feature-engineered statistical models, falling under the rubric of Machine Learning (ML) which primarily include Naive Bayes approaches \cite{mccallum1998comparison}, Logistic Regression models, \cite{genkin01082007}, and Support Vector Machines \cite{joachims1998text}. The emergence of deep neural architectures and more powerful processors designed specifically for ML and AI type computations facilitated the advanc...
- ...\textbf{Machine Learning Classifiers (ML):} This module includes classical ML algorithms frequently used in text classification tasks: Support Vector Machines (SVM) \cite{joachims1998text}, Logistic Regression (LR) \cite{genkin01082007}, and Naive Bayes (NB) \cite{mccallum1998comparison}. These classifiers operate directly on vectorized...
- ...his work, follows a similar trajectory as language representation described above, with early solutions leveraging various forms of text vectorization like TF-IDF \cite{sparckjones1972statistical} and feature reduction techniques such as LSA/LSI \cite{deerwester} used in tandem with classical Machine Learning models such as Support Vector Machines \cite{joachims1998text}, or simpler methods such as Logistic Regression or Naive Bayes \cite{mccallum1998comparison}. A good summary of the various ML approaches to text cla...

## mccallum1998comparison

```bibtex
@techreport{mccallum1998comparison,
  title={A Comparison of Event Models for Naive Bayes Text Classification},
  author={McCallum, Andrew and Nigam, Kamal},
  year={1998},
  institution={Carnegie Mellon University},
  note={Technical Report, CMU-CS-98-137},
  url={http://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf}
}
```

Citing contexts in the V6 manuscript:
- ...ning time and computational requirements (number of GPUs or CPUs, amount of memory, etc.), incurred to achieve these performance gains. Traditionally, the problem of text classification has been approached through feature-engineered statistical models, falling under the rubric of Machine Learning (ML) which primarily include Naive Bayes approaches \cite{mccallum1998comparison}, Logistic Regression models, \cite{genkin01082007}, and Support Vector Machines \cite{joachims1998text}. The emergence of deep neural architectures a...
- ...\textbf{Machine Learning Classifiers (ML):} This module includes classical ML algorithms frequently used in text classification tasks: Support Vector Machines (SVM) \cite{joachims1998text}, Logistic Regression (LR) \cite{genkin01082007}, and Naive Bayes (NB) \cite{mccallum1998comparison}. These classifiers operate directly on vectorized text features, typically using bag-of-words representations such as TF-IDF or Count vectors, and se...
- ... early solutions leveraging various forms of text vectorization like TF-IDF \cite{sparckjones1972statistical} and feature reduction techniques such as LSA/LSI \cite{deerwester} used in tandem with classical Machine Learning models such as Support Vector Machines \cite{joachims1998text}, or simpler methods such as Logistic Regression or Naive Bayes \cite{mccallum1998comparison}. A good summary of the various ML approaches to text classification can be found in the 2002 work by Fabrizio Sebastiani \cite{sebastiani2002machine}...

## sebastiani2002machine

```bibtex
@article{sebastiani2002machine,
  title={Machine learning in automated text categorization},
  author={Sebastiani, Fabrizio},
  journal={ACM Computing Surveys},
  volume={34},
  number={1},
  pages={1--47},
  year={2002},
  month={March},
  doi={10.1145/505282.505283},
  url={https://doi.org/10.1145/505282.505283}
}
```

Citing contexts in the V6 manuscript:
- ...LSA/LSI \cite{deerwester} used in tandem with classical Machine Learning models such as Support Vector Machines \cite{joachims1998text}, or simpler methods such as Logistic Regression or Naive Bayes \cite{mccallum1998comparison}. A good summary of the various ML approaches to text classification can be found in the 2002 work by Fabrizio Sebastiani \cite{sebastiani2002machine}, a paper which surveys different Machine Learning approaches to the problem of text classification before the proliferation of neural networks and de...

## zhang2015character

```bibtex
@inproceedings{zhang2015character,
  title={Character-level Convolutional Networks for Text Classification},
  author={Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  pages={649--657},
  year={2015},
  address={Montreal, Canada},
  publisher={MIT Press}
}
```

Citing contexts in the V6 manuscript:
- ...es and more powerful processors designed specifically for ML and AI type computations facilitated the advance of word embedding language models such as Word2Vec \cite{mikolov}, GloVe \cite{pennington}, and FastText \cite{articleboja} which were then used in DL models to push the boundaries for many NLP tasks, text classification being no exception \cite{zhang2015character} \cite{yoon2014convolutional} \cite{hochreiterlong}. Yet it was the introduction of transformer architecture in 2017 \cite{vaswani} followed shortly t...
- ...orks came the introduction of (static) word embedding language models which were trained in this setting as well as the introduction of various neural network architectures that were proven to be effective with text classification problems specifically - in particular work related to Convolutional Neural Networks (CNN) \cite{yoon2014convolutional} \cite{zhang2015character}, Long Short-Term Memory networks (LSTM) \cite{zang2021research}, and Attention based neural networks (ATTN) \cite{liuguo2019bidirectional}, each of w...

## zang2021research

```bibtex
@inproceedings{zang2021research,
  title={Research on Text Classification Method Based on LSTM Neural Network Model},
  author={Zang, Yifan},
  booktitle={2021 13th International Conference on Measuring Technology and Mechatronics Automation (ICMTMA)},
  pages={1019--1022},
  year={2021},
  publisher={IEEE},
  address={Dalian, China},
  doi={10.1109/ICMTMA52658.2021.00227}
}
```

Citing contexts in the V6 manuscript:
- ...els which were trained in this setting as well as the introduction of various neural network architectures that were proven to be effective with text classification problems specifically - in particular work related to Convolutional Neural Networks (CNN) \cite{yoon2014convolutional} \cite{zhang2015character}, Long Short-Term Memory networks (LSTM) \cite{zang2021research}, and Attention based neural networks (ATTN) \cite{liuguo2019bidirectional}, each of which is used as benchmark classifiers with Layer Cake to test di...

## yoon2014convolutional

```bibtex
@inproceedings{yoon2014convolutional,
  title={Convolutional Neural Networks for Sentence Classification},
  author={Kim, Yoon},
  booktitle={Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={1746--1751},
  year={2014},
  address={Doha, Qatar},
  publisher={Association for Computational Linguistics},
  url={https://aclanthology.org/D14-1181}
}
```

Citing contexts in the V6 manuscript:
- ...ssors designed specifically for ML and AI type computations facilitated the advance of word embedding language models such as Word2Vec \cite{mikolov}, GloVe \cite{pennington}, and FastText \cite{articleboja} which were then used in DL models to push the boundaries for many NLP tasks, text classification being no exception \cite{zhang2015character} \cite{yoon2014convolutional} \cite{hochreiterlong}. Yet it was the introduction of transformer architecture in 2017 \cite{vaswani} followed shortly thereafter by the introduction...
- ...age pre-trained static word embedding language models such as Word2Vec \cite{mikolov}, GloVe \cite{pennington} and FastText \cite{articleboja} models. We include three different types of neural classifiers in this module, each of which is constructed using Python PyTorch libraries\footnote{https://pytorch.org/}: Convolutional Neural Networks (CNN) \cite{yoon2014convolutional}, Attention-based models (ATTN) \cite{liuguo2019bidirectional}, and Long Short-Term Memory networks (LSTM) \cite{hochreiterlong}. The static embedding...
- ...deep learning and neural networks came the introduction of (static) word embedding language models which were trained in this setting as well as the introduction of various neural network architectures that were proven to be effective with text classification problems specifically - in particular work related to Convolutional Neural Networks (CNN) \cite{yoon2014convolutional} \cite{zhang2015character}, Long Short-Term Memory networks (LSTM) \cite{zang2021research}, and Attention based neural networks (ATTN) \cite{liuguo201...

## liuguo2019bidirectional

```bibtex
@article{liuguo2019bidirectional,
  title={Bidirectional LSTM with attention mechanism and convolutional layer for text classification},
  author={Liu, Guoyao and Guo, Jie},
  journal={Neurocomputing},
  volume={337},
  pages={325--338},
  year={2019},
  month={March},
  doi={10.1016/j.neucom.2019.01.078},
  url={https://doi.org/10.1016/j.neucom.2019.01.078}
}
```

Citing contexts in the V6 manuscript:
- ...s Word2Vec \cite{mikolov}, GloVe \cite{pennington} and FastText \cite{articleboja} models. We include three different types of neural classifiers in this module, each of which is constructed using Python PyTorch libraries\footnote{https://pytorch.org/}: Convolutional Neural Networks (CNN) \cite{yoon2014convolutional}, Attention-based models (ATTN) \cite{liuguo2019bidirectional}, and Long Short-Term Memory networks (LSTM) \cite{hochreiterlong}. The static embeddings provide fixed semantic representations of words (and subword...
- ...f various neural network architectures that were proven to be effective with text classification problems specifically - in particular work related to Convolutional Neural Networks (CNN) \cite{yoon2014convolutional} \cite{zhang2015character}, Long Short-Term Memory networks (LSTM) \cite{zang2021research}, and Attention based neural networks (ATTN) \cite{liuguo2019bidirectional}, each of which is used as benchmark classifiers with Layer Cake to test different static word embedding language models (Word2Vec, GloVe and Fasttext...

## liu2019roberta

```bibtex
@article{liu2019roberta,
  title={RoBERTa: A Robustly Optimized BERT Pretraining Approach},
  author={Liu, Yinhan and Ott, Myle and Goyal, Naman and Du, Jingfei and Joshi, Mandar and Chen, Daniel and Levy, Omer and Lewis, Mike and Zettlemoyer, Luke and others},
  journal={arXiv preprint arXiv:1907.11692},
  year={2019},
  doi={10.48550/arXiv.1907.11692},
  url={https://doi.org/10.48550/arXiv.1907.11692}
}
```

Citing contexts in the V6 manuscript:
- ...oken prediction}\footnote{Next token prediction based LLMs are called \textit{causal} language models due to the determinative nature of the underlying algorithm.} which is used for example by the GPT family of models built by OpenAI \cite{radford2019language}, and \textit{masking} which is used by the BERT \cite{devlin-etal-2019-bert} and RoBERTa \cite{liu2019roberta} based language model families models. All of these transformer-based LLMs however ultimately leverage these static, pre-trained token embeddings for ...
- ...anguage models we support, models we refer to as 'context-aware' given their support for attention masking and positional encoding information, include language models from the BERT, RoBERTa, DistilBERT (distilbert-base-cased), XLNet, GPT-2, Llama, and DeepSeek families (see Table 2 above for details). BERT \cite{devlin-etal-2019-bert} and RoBERTa \cite{liu2019roberta} are pretrained using masked language modeling (MLM), with BERT also using next sentence prediction (NSP). RoBERTa modifies BERT by removing NSP and a...
- ...iers:} The final module incorporates transformer-based classifiers using the Hugging Face Transformers library\footnote{https://huggingface.co/docs/transformers/en/index}. We support two primary architectures: the standard Sequence Classification model, which utilizes pretrained transformer models such as BERT \cite{devlin-etal-2019-bert}, RoBERTa \cite{liu2019roberta}, DistilBERT \cite{sanh2020distilbert}, XLNet \cite{yang2019xlnet}, GPT-2 \cite{radford2019language}, Llama \cite{touvron2023llama}, and DeepSeek \cit...
- ...er tuning applied to the text classification problem focused exclusively on LLMs published in late 2024 \cite{trust2024study}. However, these studies do not include classical ML approaches to text classification in their work, and while there do exist some notable text classification research surveys that do include both ML and DL approaches, e.g. \cite{liu2019roberta}, these studies are dated and do not account for the increased size and compute requirements for the more modern LLMs, properties that are becoming of...
- ...hat were shown to be effective for text classification problems were BERT, which stands for Bidirectional Encoder Representations from Transformers which was released in 2019 \cite{devlin-etal-2019-bert}, as well as derivations thereof: like RoBERTa, or Robustly Optimized BERT Pretraining Approach which was released shortly after BERT also in 2019 \cite{liu2019roberta}, as well as DistilBERT, a \textit{distilled} \cite{hinton2015distilling} variant of BERT which while smaller and more efficient computationally, was ...

## hinton2015distilling

```bibtex
@article{hinton2015distilling,
  title={Distilling the Knowledge in a Neural Network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  journal={arXiv preprint arXiv:1503.02531},
  year={2015},
  month={March},
  doi={10.48550/arXiv.1503.02531},
  url={https://doi.org/10.48550/arXiv.1503.02531}
}
```

Citing contexts in the V6 manuscript:
- ...ite{devlin-etal-2019-bert} and RoBERTa \cite{liu2019roberta} are pretrained using masked language modeling (MLM), with BERT also using next sentence prediction (NSP). RoBERTa modifies BERT by removing NSP and adjusting pretraining strategies to improve generalization. DistilBERT \cite{sanh2020distilbert}, based on BERT, uses knowledge distillation \cite{hinton2015distilling} to create a lighter, faster model while retaining most of BERT's performance. XLNet \cite{yang2019xlnet} builds on Transformer-XL \cite{dai2019transf...
- ...lex NLP tasks \cite{articlebi, guo2025deepseek}. DeepSeek models utilize reinforcement learning without preliminary SFT \cite{ouyang2022training} and incorporate chain-of-thought reasoning strategies \cite{articlewei}. In our evaluation, we also include a distilled DeepSeek model based on techniques similar to those used for optimizing Qwen models \cite{articlebai, hinton2015distilling}....
- ...e BERT, which stands for Bidirectional Encoder Representations from Transformers which was released in 2019 \cite{devlin-etal-2019-bert}, as well as derivations thereof: like RoBERTa, or Robustly Optimized BERT Pretraining Approach which was released shortly after BERT also in 2019 \cite{liu2019roberta}, as well as DistilBERT, a \textit{distilled} \cite{hinton2015distilling} variant of BERT which while smaller and more efficient computationally, was shown to be (and confirmed in this study) just as effective \cite{sanh202...

## sanh2020distilbert

```bibtex
@article{sanh2020distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  journal={arXiv preprint arXiv:1910.01108},
  year={2020},
  doi={10.48550/arXiv.1910.01108},
  url={https://doi.org/10.48550/arXiv.1910.01108}
}
```

Citing contexts in the V6 manuscript:
- ... Llama, and DeepSeek families (see Table 2 above for details). BERT \cite{devlin-etal-2019-bert} and RoBERTa \cite{liu2019roberta} are pretrained using masked language modeling (MLM), with BERT also using next sentence prediction (NSP). RoBERTa modifies BERT by removing NSP and adjusting pretraining strategies to improve generalization. DistilBERT \cite{sanh2020distilbert}, based on BERT, uses knowledge distillation \cite{hinton2015distilling} to create a lighter, faster model while retaining most of BERT's performance....
- ...es transformer-based classifiers using the Hugging Face Transformers library\footnote{https://huggingface.co/docs/transformers/en/index}. We support two primary architectures: the standard Sequence Classification model, which utilizes pretrained transformer models such as BERT \cite{devlin-etal-2019-bert}, RoBERTa \cite{liu2019roberta}, DistilBERT \cite{sanh2020distilbert}, XLNet \cite{yang2019xlnet}, GPT-2 \cite{radford2019language}, Llama \cite{touvron2023llama}, and DeepSeek \cite{guo2025deepseek}; and a transformer-...
- ...reof: like RoBERTa, or Robustly Optimized BERT Pretraining Approach which was released shortly after BERT also in 2019 \cite{liu2019roberta}, as well as DistilBERT, a \textit{distilled} \cite{hinton2015distilling} variant of BERT which while smaller and more efficient computationally, was shown to be (and confirmed in this study) just as effective \cite{sanh2020distilbert} for text classification. Figure~\ref{fig5} below shows a summary timeline of the major research milestones for Linguistics, NLP, AI and Computational...

## touvron2023llama

```bibtex
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Minervini, Paolo and Pappagari, Vamsi and others},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023},
  doi={10.48550/arXiv.2302.13971},
  url={https://doi.org/10.48550/arXiv.2302.13971}
}
```

Citing contexts in the V6 manuscript:
- ...Llama, introduced by Meta in 2023 \cite{touvron2023llama}, is a decoder-only transformer, an autoregressively trained LLM optimized for efficient scaling and multilingual generation tasks. These models are p...
- ...uggingface.co/docs/transformers/en/index}. We support two primary architectures: the standard Sequence Classification model, which utilizes pretrained transformer models such as BERT \cite{devlin-etal-2019-bert}, RoBERTa \cite{liu2019roberta}, DistilBERT \cite{sanh2020distilbert}, XLNet \cite{yang2019xlnet}, GPT-2 \cite{radford2019language}, Llama \cite{touvron2023llama}, and DeepSeek \cite{guo2025deepseek}; and a transformer-based CNN architecture, integrating transformers' context-aware embeddings with convolutional...
- ...These transformer models, each with its own nuanced training approach and transformer architecture, along with the GPT-3 model \cite{articlebrown}, as well as an XLNet model \cite{yang2019xlnet}, along with smaller language models from both the Llama \cite{touvron2023llama} and DeepSeek family \cite{articlebi}, were included in the benchmarking that was performed in this research, each of which was tested across a variet...

## qiu2020pre

```bibtex
@article{qiu2020pre,
  title={Pre-trained models for natural language processing: A survey},
  author={Qiu, Xipeng and Zhang, Yao and Zhou, Yanyan and Liu, Xuan and Sun, Xuemin},
  journal={Science China Technological Sciences},
  volume={63},
  number={10},
  pages={1872--1897},
  year={2020},
  month={September},
  doi={10.1007/s11431-020-1647-3},
  url={https://doi.org/10.1007/s11431-020-1647-3}
}
```

Citing contexts in the V6 manuscript:
- ...mmon, and very effective, means of model training and construction. While there have been a variety of good research surveys written that look at the problem of text classification within the context of different DL architectures and/or LLMs - for example relatively early studies looking at pre-trained language models for NLP more broadly in 2020, \cite{qiu2020pre}, another focused on deep learning approaches for text classification specifically \cite{minaee2021deep}, and another very thorough study that looks a...

## minaee2021deep

```bibtex
@article{minaee2021deep,
  title={Deep Learning-based Text Classification: A Comprehensive Review},
  author={Minaee, Shervin and Kalchbrenner, Nal and Yang, Wen and Ghanbari, Hadi and others},
  journal={ACM Computing Surveys (CSUR)},
  volume={54},
  number={3},
  pages={1--40},
  year={2021},
  month={April},
  doi={10.1145/3439726},
  url={https://doi.org/10.1145/3439726}
}
```

Citing contexts in the V6 manuscript:
- ...of good research surveys written that look at the problem of text classification within the context of different DL architectures and/or LLMs - for example relatively early studies looking at pre-trained language models for NLP more broadly in 2020, \cite{qiu2020pre}, another focused on deep learning approaches for text classification specifically \cite{minaee2021deep}, and another very thorough study that looks at the effectiveness of various optimization techniques and hyperparameter tuning applied to the text cla...

## trust2024study

```bibtex
@article{trust2024study,
  title={A Study on Text Classification in the Age of Large Language Models},
  author={Trust, Peter and Minghim, Rafael},
  journal={Machine Learning and Knowledge Extraction},
  volume={6},
  number={4},
  pages={2688--2721},
  year={2024},
  month={November},
  doi={10.3390/make6040129},
  url={https://doi.org/10.3390/make6040129}
}
```

Citing contexts in the V6 manuscript:
- ...n 2020, \cite{qiu2020pre}, another focused on deep learning approaches for text classification specifically \cite{minaee2021deep}, and another very thorough study that looks at the effectiveness of various optimization techniques and hyperparameter tuning applied to the text classification problem focused exclusively on LLMs published in late 2024 \cite{trust2024study}. However, these studies do not include classical ML approaches to text classification in their work, and while there do exist some notable text class...

## dai2019transformer

```bibtex
@article{dai2019transformer,
  title={Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context},
  author={Dai, Zihang and Yang, Zhilin and Yang, Yiming and Carbonell, Jaime and Le, Quoc V. and Salakhutdinov, Ruslan},
  journal={arXiv preprint arXiv:1901.02860},
  year={2019},
  month={June},
  doi={10.48550/arXiv.1901.02860},
  url={https://doi.org/10.48550/arXiv.1901.02860}
}
```

Citing contexts in the V6 manuscript:
- ...n (NSP). RoBERTa modifies BERT by removing NSP and adjusting pretraining strategies to improve generalization. DistilBERT \cite{sanh2020distilbert}, based on BERT, uses knowledge distillation \cite{hinton2015distilling} to create a lighter, faster model while retaining most of BERT's performance. XLNet \cite{yang2019xlnet} builds on Transformer-XL \cite{dai2019transformer} and introduces permutation-based training to capture bidirectional context without masking, blending ideas from both encoder and autoregressive archi...

## guo2025deepseek

```bibtex
@article{guo2025deepseek,
  title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author={Guo, Dong and Zhang, Shilin and Liu, Zizhen and Wu, Jiayi and Wang, Haotian and Wang, Yang and Chen, Xiangyu and Jin, Xin and Wang, Haitao and Liu, Yizhe and others},
  journal={arXiv preprint arXiv:2501.12948},
  year={2025},
  month={January},
  doi={10.48550/arXiv.2501.12948},
  url={https://doi.org/10.48550/arXiv.2501.12948}
}
```

Citing contexts in the V6 manuscript:
- ...s and are also tuned using supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) methods. We also include one of the smaller DeepSeek models in our study. This family of transformer-based models are known for their efficiency and reasoning capabilities, and have achieved strong results on a variety of complex NLP tasks \cite{articlebi, guo2025deepseek}. DeepSeek models utilize reinforcement learning without preliminary SFT \cite{ouyang2022training} and incorporate chain-of-thought reasoning strategi...
- ...ex}. We support two primary architectures: the standard Sequence Classification model, which utilizes pretrained transformer models such as BERT \cite{devlin-etal-2019-bert}, RoBERTa \cite{liu2019roberta}, DistilBERT \cite{sanh2020distilbert}, XLNet \cite{yang2019xlnet}, GPT-2 \cite{radford2019language}, Llama \cite{touvron2023llama}, and DeepSeek \cite{guo2025deepseek}; and a transformer-based CNN architecture, integrating transformers' context-aware embeddings with convolutional layers to enhance feature extraction...

## ouyang2022training

```bibtex
@article{ouyang2022training,
  title={Training language models to follow instructions with human feedback},
  author={Ouyang, Long and Wu, Jeff and Jiang, Xu and Almeida, Diogo and Wainwright, Carroll and Mishkin, Pamela and Zhang, Chong and Agarwal, Sandhini and Slama, Katarina and Ray, Alex and others},
  journal={arXiv preprint arXiv:2203.02155},
  year={2022},
  month={March},
  doi={10.48550/arXiv.2203.02155},
  url={https://doi.org/10.48550/arXiv.2203.02155}
}
```

Citing contexts in the V6 manuscript:
- ...HF) methods. We also include one of the smaller DeepSeek models in our study. This family of transformer-based models are known for their efficiency and reasoning capabilities, and have achieved strong results on a variety of complex NLP tasks \cite{articlebi, guo2025deepseek}. DeepSeek models utilize reinforcement learning without preliminary SFT \cite{ouyang2022training} and incorporate chain-of-thought reasoning strategies \cite{articlewei}. In our evaluation, we also include a distilled DeepSeek model based on techn...

## lewis1987reuters

```bibtex
@misc{lewis1987reuters,
  title ={ Reuters-21578 Text Categorization Collection [Dataset]},
  author = {Lewis, David D.},
  year = {1987},
  url = {https://archive.ics.uci.edu/ml/datasets/reuters-21578+text+categorization+collection},
  doi = {https://doi.org/10.24432/C52G6M}
}
```

Citing contexts in the V6 manuscript:
- ...\begin{itemize} \item \textbf{Reuters-21578} \cite{lewis1987reuters}: A multi-label dataset containing 21,578 news documents across 115 categories. Although class imbalance is significant, it remains a standard benchma...

## greene

```bibtex
@inproceedings{greene,
	author = {Greene, Derek and Cunningham, Padraig},
	year = {2006},
	month = {06},
	isbn = {1595933832},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	url = {https://doi.org/10.1145/1143844.1143892},
	booktitle = {Proceedings of the 23rd International Conference on Machine Learning},
	pages = {377–384},
	numpages = {8},
	location = {Pittsburgh, Pennsylvania, USA},
	series = {ICML '06}
	title = {Practical solutions to the problem of diagonal dominance in kernel document clustering},
	volume = {148},
	journal = {ACM International Conference Proceeding Series},
	doi = {10.1145/1143844.1143892}
}
```

Citing contexts in the V6 manuscript:
- ... \item \textbf{BBC News} \cite{greene}: A single-label dataset of 2,225 news articles categorized into five balanced topics: business, entertainment, politics, sport, and tech....

## lewis2004

```bibtex
@article{lewis2004,
	author = {Lewis, David and Yang, Yiming and Russell-Rose, Tony and Li, Fan},
	year = {2004},
	month = {04},
	pages = {361-397},
	title = {RCV1: A New Benchmark Collection for Text Categorization Research.},
	volume = {5},
	journal = {Journal of Machine Learning Research}
}
```

Citing contexts in the V6 manuscript:
- ... \item \textbf{RCV1-V2} \cite{lewis2004}: A large-scale multi-label dataset with over 800,000 Reuters newswire stories spanning 103 overlapping categories, ideal for evaluating models on com...
- ... datasets is listed in the table below, and is broken down by classifier as each classifier has slightly different preprocessing logic, with the neural networks requiring a validation data set set aside from the training data to test the training epochs. Note that we, following \cite{moreo2021}, use the standard 'Lewis Split' for the RCV-1 dataset \cite{lewis2004}, allowing us to evaluate our results more precisely against prior research....

## LANG1995331

```bibtex
@incollection{LANG1995331,
title = {NewsWeeder: Learning to Filter Netnews},
editor = {Armand Prieditis and Stuart Russell},
booktitle = {Machine Learning Proceedings 1995},
publisher = {Morgan Kaufmann},
address = {San Francisco (CA)},
pages = {331-339},
year = {1995},
isbn = {978-1-55860-377-6},
doi = {https://doi.org/10.1016/B978-1-55860-377-6.50048-7},
url = {https://www.sciencedirect.com/science/article/pii/B9781558603776500487},
author = {Ken Lang},
}
```

Citing contexts in the V6 manuscript:
- ... \item \textbf{20 Newsgroups} \cite{LANG1995331}: A well-known single-label dataset containing approximately 20,000 documents across 20 categories, widely used for benchmarking text classification a...

## sinha

```bibtex
@inproceedings{sinha,
author = {Sinha, Arnab and Shen, Zhihong and Song, Yang and Ma, Hao and Eide, Darrin and Hsu, Bo-June and Wang, Kuansan},
year = {2015},
month = {05},
pages = {243-246},
title = {An Overview of Microsoft Academic Service (MAS) and Applications},
doi = {10.1145/2740908.2742839}
}
```

Citing contexts in the V6 manuscript:
- ... \item \textbf{arXiv} \cite{sinha}: A multi-label dataset of research papers from the arXiv repository, covering 58 scientific disciplines. It tests model robustness on technical, doma...

## farhangi

```bibtex
@inproceedings{farhangi,
author = {Farhangi, Ashkan and Sui, Ning and Hua, Nan and Bai, Haiyan and Huang, Arthur and Guo, Zhishan},
year = {2022},
month = {06},
pages = {447-458},
title = {Protoformer: Embedding Prototypes for Transformers},
isbn = {978-3-031-05932-2},
doi = {10.1007/978-3-031-05933-9_35}
}
```

Citing contexts in the V6 manuscript:
- ... \item \textbf{arXiv protoformer} \cite{farhangi}: A simplified, single-label subset of the arXiv dataset focused on the 10 most common scientific categories, frequently used for prototype-based mode...

## sanderson

```bibtex
@article{sanderson,
author = {Sanderson, Mark},
year = {2010},
month = {01},
pages = {247-375},
title = {Test Collection Based Evaluation of Information Retrieval Systems},
volume = {4},
journal = {Foundations and Trends in Information Retrieval},
doi = {10.1561/1500000009}
}
```

Citing contexts in the V6 manuscript:
- ... \item \textbf{OHSUMED} \cite{sanderson}: A multi-label dataset comprising medical abstracts labeled with 23 MeSH (Medical Subject Headings) classes, presenting challenges due to its highly ...

## maas

```bibtex
@inproceedings{maas,
author = {Maas, Andrew and Daly, Raymond and Pham, Peter and Huang, Dan and Ng, Andrew and Potts, Christopher},
year = {2011},
month = {01},
pages = {142-150},
title = {Learning Word Vectors for Sentiment Analysis}
}
```

Citing contexts in the V6 manuscript:
- ... \item \textbf{IMDB} \cite{maas}: A binary classification dataset of movie reviews labeled as positive or negative, balanced across training and test splits. The significant variance...

## tifrea

```bibtex
@misc{tifrea,
	author = {Tifrea, Alexandru and Becigneul, Gary and Ganea, Octavian},
	year = {2018},
	month = {10},
	pages = {},
	title = {Poincar\'e GloVe: Hyperbolic Word Embeddings},
	doi = {10.48550/arXiv.1810.06546}
}
```

Citing contexts in the V6 manuscript:
- ...e testing we've done with Layer Cake covers a broad spectrum of classifiers and language model forms, we nonetheless limit the study to Euclidean spatial geometry, i.e. all of the language representation forms we work with are constructed in Euclidean spaces. However, there is a lot of promising research that has been done with hyperbolic geometry \cite{tifrea} \cite{leimeister2019skip} \cite{valentino} and it would be very interesting to see how these embedding forms performed relatively speaking, or to see...

## leimeister2019skip

```bibtex
@article{leimeister2019skip,
  title={Skip-gram word embeddings in hyperbolic space},
  author={Leimeister, Matthias and Wilson, Benjamin J.},
  journal={arXiv preprint arXiv:1905.09791},
  year={2019},
  month={May},
  url={https://arxiv.org/abs/1905.09791}
}
```

Citing contexts in the V6 manuscript:
- ...e done with Layer Cake covers a broad spectrum of classifiers and language model forms, we nonetheless limit the study to Euclidean spatial geometry, i.e. all of the language representation forms we work with are constructed in Euclidean spaces. However, there is a lot of promising research that has been done with hyperbolic geometry \cite{tifrea} \cite{leimeister2019skip} \cite{valentino} and it would be very interesting to see how these embedding forms performed relatively speaking, or to see if using these embeddings...

## valentino

```bibtex
@misc{valentino,
	author = {Valentino, Marco and Carvalho, Danilo and Freitas, Andre},
	year = {2023},
	month = {05},
	pages = {},
	title = {Multi-Relational Hyperbolic Word Embeddings from Natural Language Definitions},
	doi = {10.48550/arXiv.2305.07303}
}
```

Citing contexts in the V6 manuscript:
- ...ers a broad spectrum of classifiers and language model forms, we nonetheless limit the study to Euclidean spatial geometry, i.e. all of the language representation forms we work with are constructed in Euclidean spaces. However, there is a lot of promising research that has been done with hyperbolic geometry \cite{tifrea} \cite{leimeister2019skip} \cite{valentino} and it would be very interesting to see how these embedding forms performed relatively speaking, or to see if using these embeddings that are trained...

