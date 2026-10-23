## articleding

```bibtex
@article{articleding,
author = {Ding, Ning and Qin, Yujia and Yang, Guang and Wei, Fuchao and Zonghan, Yang and Su, Yusheng and Hu, Shengding and Chen, Yulin and Chan, Chi-Min and Chen, Weize and Yi, Jing and Zhao, Weilin and Wang, Xiaozhi and Liu, Zhiyuan and Zheng, Hai-Tao and Chen, Jianfei and Liu, Yang and Tang, Jie and Li, Juanzi and Sun, Maosong},
year = {2023},
month = {03},
pages = {1-16},
title = {Parameter-efficient fine-tuning of large-scale pre-trained language models},
volume = {5},
journal = {Nature Machine Intelligence},
doi = {10.1038/s42256-023-00626-4}
}
```

Citing contexts in the V6 manuscript:
- ...Also from a performance perspective, it would be informative to further evaluate various parameterization tuning methods, for example as described \cite{lee} or in the more comprehensive look at parameter efficient fine-tuning methods (PEFT) from 2023 \cite{articleding}. In our tests, we did not find the language models, specifically the transformer-based classifiers, to be performant when freezing only a certain num...

## articlebai

```bibtex
@article{articlebai,
author = {Bai, Jinze and Bai, Shuai and Chu, Yunfei and Cui, Zeyu and Dang, Kai and Deng, Xiaodong and Fan, Yang and Ge, Wenbin and Han, Yu and Huang, Fei and Hui, Binyuan and Ji, Luo and Li, Mei and Lin, Junyang and Lin, Runji and Liu, Dayiheng and Liu, Gao and Lu, Chengqiang and Lu, Keming and Zhou, Xiaohuan},
year = {2023},
month = {09},
pages = {},
title = {QWEN TECHNICAL REPORT}
}
```

Citing contexts in the V6 manuscript:
- ...lex NLP tasks \cite{articlebi, guo2025deepseek}. DeepSeek models utilize reinforcement learning without preliminary SFT \cite{ouyang2022training} and incorporate chain-of-thought reasoning strategies \cite{articlewei}. In our evaluation, we also include a distilled DeepSeek model based on techniques similar to those used for optimizing Qwen models \cite{articlebai, hinton2015distilling}....

## articlebi

```bibtex
@article{articlebi,
author = {Bi, Xiao and Chen, Deli and Chen, Guanting and Chen, Shanhuang and Dai, Damai and Deng, Chengqi and Ding, Honghui and Dong, Kai and Du, Qiushi and Fu, Zhe and Gao, Huazuo and Gao, Kaige and Gao, Wenjun and Ge, Ruiqi and Guan, Kang and Guo, Daya and Guo, Jianzhong and Hao, Guangbo and Hao, Zhewen and Zou, Yuheng},
year = {2024},
month = {01},
pages = {},
title = {DeepSeek LLM Scaling Open-Source Language Models with Longtermism}
}
```

Citing contexts in the V6 manuscript:
- ...s and are also tuned using supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) methods. We also include one of the smaller DeepSeek models in our study. This family of transformer-based models are known for their efficiency and reasoning capabilities, and have achieved strong results on a variety of complex NLP tasks \cite{articlebi, guo2025deepseek}. DeepSeek models utilize reinforcement learning without preliminary SFT \cite{ouyang2022training} and incorporate chain-of-thought reasoning strategi...
- ...These transformer models, each with its own nuanced training approach and transformer architecture, along with the GPT-3 model \cite{articlebrown}, as well as an XLNet model \cite{yang2019xlnet}, along with smaller language models from both the Llama \cite{touvron2023llama} and DeepSeek family \cite{articlebi}, were included in the benchmarking that was performed in this research, each of which was tested across a variety of classifiers and each of which wa...

## articleboja

```bibtex
@article{articleboja,
author = {Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
year = {2016},
month = {07},
pages = {},
title = {Enriching Word Vectors with Subword Information},
volume = {5},
journal = {Transactions of the Association for Computational Linguistics},
doi = {10.1162/tacl_a_00051}
}
```

Citing contexts in the V6 manuscript:
- ...Regression models, \cite{genkin01082007}, and Support Vector Machines \cite{joachims1998text}. The emergence of deep neural architectures and more powerful processors designed specifically for ML and AI type computations facilitated the advance of word embedding language models such as Word2Vec \cite{mikolov}, GloVe \cite{pennington}, and FastText \cite{articleboja} which were then used in DL models to push the boundaries for many NLP tasks, text classification being no exception \cite{zhang2015character} \cite{y...
- ...pennington} we use is glove.840B.300d.txt which captures global co-occurrence statistics across a large corpus, providing 300-dimensional embeddings. The Word2Vec model \cite{mikolov}, GoogleNews-vectors-negative300.bin, was trained on Google News data using CBOW and skip-gram methods with negative sampling for efficient learning, and the FastText \cite{articleboja} model we use is crawl-300d-2M.vec.bin, which extends Word2Vec by incorporating subword information, enabling generalization to out-of-vocabulary word...
- ...\textbf{Static Language Model Classifiers (Static LM):} The second module employs neural network-based classifiers, originally presented in \cite{moreo2021}, which are designed to leverage pre-trained static word embedding language models such as Word2Vec \cite{mikolov}, GloVe \cite{pennington} and FastText \cite{articleboja} models. We include three different types of neural classifiers in this module, each of which is constructed using Python PyTorch libraries\footnote{h...
- ...FastText on the other hand, which first introduced in 2017 by Meta's AI team \cite{articleboja}, while very much related to Word2Vec in the sense that it is also a local co-occurrence computed (static) word embedding model, it is unique in that ...

## articlebrown

```bibtex
@article{articlebrown,
author = {Brown, Tom and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam, Pranav and Sastry, Girish and Askell, Amanda and Agarwal, Sandhini and Herbert-Voss, Ariel and Krueger, Gretchen and Henighan, Tom and Child, Rewon and Ramesh, Aditya and Ziegler, Daniel and Wu, Jeffrey and Winter, Clemens and Amodei, Dario},
year = {2020},
month = {05},
pages = {},
title = {Language Models are Few-Shot Learners},
doi = {10.48550/arXiv.2005.14165}
}
```

Citing contexts in the V6 manuscript:
- ...These transformer models, each with its own nuanced training approach and transformer architecture, along with the GPT-3 model \cite{articlebrown}, as well as an XLNet model \cite{yang2019xlnet}, along with smaller language models from both the Llama \cite{touvron2023llama} and DeepSeek family \...

## articlekatz

```bibtex
@article{articlekatz,
author = {Katz, Daniel and Bommarito, Michael and Gao, Shang and Arredondo, Pablo},
year = {2024},
month = {02},
pages = {},
title = {GPT-4 passes the bar exam},
volume = {382},
journal = {Philosophical Transactions of the Royal Society A},
doi = {10.1098/rsta.2023.0254}
}
```

Citing contexts in the V6 manuscript:
- (not cited in V6)

## articlejones

```bibtex
@article{articlejones,
  title={Large Language Models Pass the Turing Test},
  author={Jones, Cameron R. and Bergen, Benjamin K.},
  journal={arXiv preprint arXiv:2503.23674},
  year={2025},
  url={https://arxiv.org/abs/2503.23674},
  note={arXiv:2503.23674 [cs.CL]}
}
```

Citing contexts in the V6 manuscript:
- (not cited in V6)

## articlewei

```bibtex
@article{articlewei,
	author = {Wei, Jason and Wang, Xuezhi and Schuurmans, Dale and Bosma, Maarten and Chi, Ed and Le, Quoc and Zhou, Denny},
	year = {2022},
	month = {01},
	pages = {},
	title = {Chain of Thought Prompting Elicits Reasoning in Large Language Models},
	doi = {10.48550/arXiv.2201.11903}
}
```

Citing contexts in the V6 manuscript:
- ...is family of transformer-based models are known for their efficiency and reasoning capabilities, and have achieved strong results on a variety of complex NLP tasks \cite{articlebi, guo2025deepseek}. DeepSeek models utilize reinforcement learning without preliminary SFT \cite{ouyang2022training} and incorporate chain-of-thought reasoning strategies \cite{articlewei}. In our evaluation, we also include a distilled DeepSeek model based on techniques similar to those used for optimizing Qwen models \cite{articlebai,...
- ...f foundational LLMs that continue to advance along many different dimensions, in particular with respect to support for multiple \textit{modalities} - i.e. pictures, text, sound, etc - as well as reasoning capabilities, sometimes referred to as \textit{chain-of-thought}.\footnote{Chain-Of-Thought, or COT, capabilities were first introduced in 2022 \cite{articlewei} but have more recently come to refer to the feature that many of the advanced language generation tools have integrated where the specific reasoning ...

## xu2025towards

```bibtex
@article{xu2025towards,
	title={Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models},
	author={Fengli Xu and Qianyue Hao and Zefang Zong and Jingwei Wang and Yunke Zhang and Jingyi Wang and Xiaochong Lan and Jiahui Gong and Tianjian Ouyang and Fanjin Meng and Chenyang Shao and Yuwei Yan and Qinglong Yang and Yiwen Song and Sijian Ren and Xinyuan Hu and Yu Li and Jie Feng and Chen Gao and Yong Li},
	year={2025},
	eprint={2501.09686},
	archivePrefix={arXiv},
	primaryClass={cs.AI},
	url={https://arxiv.org/abs/2501.09686}
}
```

Citing contexts in the V6 manuscript:
- (not cited in V6)

## chomsky56

```bibtex
@article{chomsky56,
  author={Chomsky, N.},
  journal={IRE Transactions on Information Theory}, 
  title={Three models for the description of language}, 
  year={1956},
  volume={2},
  number={3},
  pages={113-124},
  keywords={Natural languages;Testing;Laboratories;Markov processes;Impedance matching;Kernel;Research and development},
  doi={10.1109/TIT.1956.1056813}}
```

Citing contexts in the V6 manuscript:
- ...morphology that were established primarily by Noam Chomsky in the middle of the last century. His most influential works with respect to computational linguistics perhaps being ''Three Models for the description of Language'', an article published in 1956 where he introduces formal grammars which now serve as the basis for computer language theory \cite{chomsky56} - one of the core tenets of theoretical computer science, from which the famed and oft cited Computer Science \textit{hierarchy of languages} is deri...

## Chomsky57

```bibtex
@book{Chomsky57,
url = {https://doi.org/10.1515/9783112316009},
title = {Syntactic Structures},
author = {Noam Chomsky},
publisher = {De Gruyter Mouton},
address = {Berlin, Boston},
doi = {doi:10.1515/9783112316009},
isbn = {9783112316009},
year = {1957},
lastchecked = {2025-04-16}
}
```

Citing contexts in the V6 manuscript:
- ...cited Computer Science \textit{hierarchy of languages} is derived (see Figure~\ref{fig3}). In the book \textit{Syntactic Structures}, published in 1957, Chomsky introduces the notion of \textit{generative grammars} which serves as the basis for the hierarchy of languages which one of the most fundamental conceptual frameworks in Computation Theory \cite{Chomsky57}. Furthermore, in the and the book "Aspects of the Theory of Syntax" which was first published in 1965, Chomsky outlines the notion of generational gr...

## chomsky65aspects

```bibtex
@book{chomsky65aspects,
  title={Aspects of the Theory of Syntax},
  author={Chomsky, Noam},
  year={1965},
  publisher={MIT Press},
  address={Cambridge, MA}
}
```

Citing contexts in the V6 manuscript:
- ...ntax" which was first published in 1965, Chomsky outlines the notion of generational grammars, distinguishing between deep and surface structures which in turn inform our modern notions of \textit{syntax} and \textit{semantics} which provide the theoretical foundations for linguistics which in turn drives much of the theoretical foundations of NLP \cite{chomsky65aspects}....

## harris

```bibtex
@article{harris,
  title={Distributional Structure},
  author={Harris, Zellig S.},
  journal={WORD},
  volume={10},
  number={2-3},
  pages={146--162},
  year={1954},
  publisher={Taylor \& Francis},
  doi={10.1080/00437956.1954.11659520},
  url={https://doi.org/10.1080/00437956.1954.11659520}
}
```

Citing contexts in the V6 manuscript:
- ...One of the most prominent and prevalent principles that is borrowed from linguistics theory which is fundamental to language model training is the \textit{distributional hypothesis}, a core tenet of semantic theory whose origins go back to middle of the last century \cite{harris} \cite{firth} and which have been summarized beautifully by J.R. Firth, one of the scholars attributed to discovering this principle, as \textit{you s...

## firth

```bibtex
@incollection{firth,
  title={A Synopsis of Linguistic Theory 1930-55},
  author={Firth, J. R.},
  booktitle={Studies in Linguistic Analysis},
  pages={1--32},
  year={1957},
  publisher={Philological Society},
  note={Special volume, 1952--59}
}
```

Citing contexts in the V6 manuscript:
- ...One of the most prominent and prevalent principles that is borrowed from linguistics theory which is fundamental to language model training is the \textit{distributional hypothesis}, a core tenet of semantic theory whose origins go back to middle of the last century \cite{harris} \cite{firth} and which have been summarized beautifully by J.R. Firth, one of the scholars attributed to discovering this principle, as \textit{you shall know a w...

## vaswani

```bibtex
@inproceedings{vaswani,
 author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, \L ukasz and Polosukhin, Illia},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {I. Guyon and U. Von Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
 pages = {6000--6010},
 publisher = {Curran Associates, Inc.},
 title = {Attention is All you Need},
 url = {https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf},
 volume = {30},
 year = {2017}
}
```

Citing contexts in the V6 manuscript:
- ...echnological advancement first introduced in 2012 \cite{mikolov} which looked to provide a more effective, and efficient, means of representing language (i.e. \textit{word embeddings}) which in turn led to the development of what have come to be known ad \textit{large language models}\textit, or LLMs, which leverage transformer-based architectures \cite{vaswani} (introduced in 2017) along with multi-head attention mechanisms in a deep learning (DL) context to construct language models that can both represent ...
- ...ntion masking features which are fundamental to transformer-based language models, which facilitate much more powerful interpretive capabilities of these LLMs which have transformed the AI and technological landscape over the past few years. This is the basic architecture that was introduced in 2017 in the seminal "Attention is all you need" paper \cite{vaswani} except described from a semantic perspective, and it is hard to exaggerate the profound effect that these technological advancements have already had...
- ...s such as Word2Vec \cite{mikolov}, GloVe \cite{pennington}, and FastText \cite{articleboja} which were then used in DL models to push the boundaries for many NLP tasks, text classification being no exception \cite{zhang2015character} \cite{yoon2014convolutional} \cite{hochreiterlong}. Yet it was the introduction of transformer architecture in 2017 \cite{vaswani} followed shortly thereafter by the introduction of BERT \cite{devlin-etal-2019-bert} which demonstrated how powerful LLMs could be in establishing ne...
- ...e granular representation of a text sequence as a function of a) the semantic meaning of the pretrained token embeddings, and b) a representation of the underlying semantics of a given text sequence, sentence or passage. This additive semantic design strategy is arguably the very core of the technology innovation introduced by transformers in 2017 \cite{vaswani}....
- ...oss that very same spectrum of NLP problems, with these advanced language generation tools simply being the most widespread and most well known examples. This is arguably the most important innovation that comes with transformers and was precisely the motivation of the \textit{attention mechanism} \cite{bahdanau2016neuralmachinetranslationjointly} \cite{vaswani} which underpins transformer-based language models which facilitates the creation of a more complex semantic representation of language by breaking la...
- ... tokens, or subwords, and then combining static representations of these tokens (embeddings) with positional information (also embeddings), in order to create what we call "context-aware" representations of language (again, embeddings). It is these advancements in language representation in fact that drive the power of the transformer architecture \cite{vaswani} which is the engine inside the hood of almost every modern LLMs in one form or another. ...
- ...s a model which allows for a better understanding of the \textit{meaning} of the specific token in that context as reflected in the dynamically generated embeddings that are created to represent that token by the transformer. This is where \textit{positional embeddings} come in, one of the most revolutionary parts of the original Transformer paper \cite{vaswani}. There are two main types of positional embeddings that are used by transformers (1) learned positional embeddings, where each position (e.g., 0-511)...

## salton1971smart

```bibtex
@book{salton1971smart,
  title={The SMART Retrieval System: Experiments in Automatic Document Processing},
  author={Salton, Gerard},
  year={1971},
  publisher={Prentice-Hall, Inc.},
  address={Englewood Cliffs, NJ}
}
```

Citing contexts in the V6 manuscript:
- ...A different form of language representation emerged in the 1970s however, one that was informed by linguistics theory that looked to project language into a semantic space to facilitate document search and information retrieval - namely \textit{vector space models} \cite{salton1971smart} which form the basis of semantic spaces of meaning which provide the mathematical, geometrical, and linguistic foundations of \textit{word embeddings...

## sparckjones1972statistical

```bibtex
@article{sparckjones1972statistical,
  title={A Statistical Interpretation of Term Specificity and Its Application in Retrieval},
  author={Sparck Jones, Karen},
  journal={Journal of Documentation},
  volume={28},
  number={1},
  pages={11--21},
  year={1972},
  doi={10.1108/eb026526},
  url={https://doi.org/10.1108/eb026526}
}
```

Citing contexts in the V6 manuscript:
- ...ques such as one hot encoding, or Count Vectorization as it is sometimes called, a clasely related cousin to the very influential and still very popular form of text vectorization which takes into account the term's 'importance' within a given corpus called 'term-frequency inverse-document-frequency', or simply TF-IDF, which was introduced in 1972 \cite{sparckjones1972statistical}. But text vectorization approaches to language representation, which are still widely used today in particular within the context of classical machin...
- ...Research related to Text Classification, one of the major themes of this work, follows a similar trajectory as language representation described above, with early solutions leveraging various forms of text vectorization like TF-IDF \cite{sparckjones1972statistical} and feature reduction techniques such as LSA/LSI \cite{deerwester} used in tandem with classical Machine Learning models such as Support Vector Machi...

## deerwester

```bibtex
@article{deerwester,
  title={Indexing by latent semantic analysis},
  author={Deerwester, Scott and Dumais, Susan T. and Furnas, George W. and Landauer, Thomas K. and Harshman, Richard},
  journal={Journal of the American Society for Information Science},
  volume={41},
  number={6},
  pages={391--407},
  year={1990},
  doi={10.1002/(SICI)1097-4571(199009)41:6<391::AID-ASI1>3.0.CO;2-9},
  url={https://doi.org/10.1002/(SICI)1097-4571(199009)41:6<391::AID-ASI1>3.0.CO;2-9}
}
```

Citing contexts in the V6 manuscript:
- ...which looks to bring down the size of the input feature matrix to ML models while retaining as much of its predictive power as possible. Common feature reduction techniques such as Principle Component Analysis (PCA) \cite{pearson01111901} \cite{hotelling1933AnalysisOA}, as well as Latent Semantic Analysis, or LSA (or Latent Semantic Indexing, LSI) \cite{deerwester}, which leverages matrix \textit{eigendecomposition} in order to identify latent contextual (effectively semantic) relationships between words or term...
- ...Research related to Text Classification, one of the major themes of this work, follows a similar trajectory as language representation described above, with early solutions leveraging various forms of text vectorization like TF-IDF \cite{sparckjones1972statistical} and feature reduction techniques such as LSA/LSI \cite{deerwester} used in tandem with classical Machine Learning models such as Support Vector Machines \cite{joachims1998text}, or simpler methods such as Logistic Re...
- ...We also include a few variants of Latent Semantic Analysis (LSA), sometimes called Latent Semantic Indexing (LSI), \cite{deerwester} into the ML representation form mix as well, a method that continues to show its value as an extremely effective model optimization technique. We eve...

## mikolov

```bibtex
@article{mikolov,
author = {Mikolov, Tomas and Chen, Kai and Corrado, G.s and Dean, Jeffrey},
year = {2013},
month = {01},
pages = {},
title = {Efficient Estimation of Word Representations in Vector Space},
volume = {2013},
journal = {Proceedings of Workshop at ICLR}
}
```

Citing contexts in the V6 manuscript:
- ...Language models are a technological advancement first introduced in 2012 \cite{mikolov} which looked to provide a more effective, and efficient, means of representing language (i.e. \textit{word embeddings}) which in turn led to the deve...
- ...es approaches \cite{mccallum1998comparison}, Logistic Regression models, \cite{genkin01082007}, and Support Vector Machines \cite{joachims1998text}. The emergence of deep neural architectures and more powerful processors designed specifically for ML and AI type computations facilitated the advance of word embedding language models such as Word2Vec \cite{mikolov}, GloVe \cite{pennington}, and FastText \cite{articleboja} which were then used in DL models to push the boundaries for many NLP tasks, text classific...
- ... trained on the Common Crawl dataset, a large-scale, multilingual web corpus consisting of over 600 billion tokens.\footnote{Source: https://commoncrawl.org/} The GloVe model \cite{pennington} we use is glove.840B.300d.txt which captures global co-occurrence statistics across a large corpus, providing 300-dimensional embeddings. The Word2Vec model \cite{mikolov}, GoogleNews-vectors-negative300.bin, was trained on Google News data using CBOW and skip-gram methods with negative sampling for efficient learning, ...
- ...\textbf{Static Language Model Classifiers (Static LM):} The second module employs neural network-based classifiers, originally presented in \cite{moreo2021}, which are designed to leverage pre-trained static word embedding language models such as Word2Vec \cite{mikolov}, GloVe \cite{pennington} and FastText \cite{articleboja} models. We include three different types of neural classifiers in this module, each of which...
- ...nformation retrieval - namely \textit{vector space models} \cite{salton1971smart} which form the basis of semantic spaces of meaning which provide the mathematical, geometrical, and linguistic foundations of \textit{word embeddings} which have emerged as the basic building blocks of LLMs. Word embeddings were first introduced in 2013 with Word2Vec \cite{mikolov} which uses DL along with word co-occurrence metrics to learn (word) embeddings by predicting words based on (local) word contexts\footnote{Window con...

## worth

```bibtex
@article{worth,
  title={Word Embeddings and Semantic Spaces in Natural Language Processing},
  author={Worth, P.},
  journal={International Journal of Intelligence Science},
  volume={13},
  number={1},
  pages={1--21},
  year={2023},
  doi={10.4236/ijis.2023.131001},
  url={https://doi.org/10.4236/ijis.2023.131001}
}
```

Citing contexts in the V6 manuscript:
- ...Architecturally speaking, each of these LLMs comes with a pre-computed set of (static) embeddings for each token in its dictionary which identify, and more precisely geometrically locate, each token in the underlying feature space, what we refer to as the underlying \textit{semantic space} \cite{worth}. These static embeddings are computed during pre-training based upon the specific training techniques used by the underlying LLM and based off of som...
- ...(i.e. a \textit{prompt}), or a text summarization output based upon a given textual input, or in our case a classification label (or probability distribution of labels) based upon a given input doc or text sequence. These embeddings serve as search vectors in the underlying pretrained Euclidean space (\textit{semantic space} effectively, following \cite{worth}, which provide rich information to the underlying model. These pretrained embedding forms of language representation of course are very different fro...
- ...looking at these different representation forms within the context of these classic machine learning classifiers is that the underlying mathematics, and the underlying forms of language representation are relatively straightforward and are coupled together directly, as a form of linear separation in vector space, in semantic space really following \cite{worth}. Once we get to the transformer classifiers, we become constrained by the underlying language model itself in terms of how the text can be represente...
- ...parameter conditions and each with its own distinct, although closely related, form of language representation. In fact, technically, each language model has its own form of language representation, albeit all architecturally analogous, all nonetheless distinct with respect to how the underlying feature space, or \textit{semantic space} is defined \cite{worth}. Part of what we are looking to understand with Layer Cake in fact is the difference this type of dimensional attribute has on classifier performance...

## pennington

```bibtex
@inproceedings{pennington,
author = {Pennington, Jeffrey and Socher, Richard and Manning, Christopher},
year = {2014},
month = {01},
pages = {1532-1543},
title = {Glove: Global Vectors for Word Representation},
volume = {14},
journal = {EMNLP},
doi = {10.3115/v1/D14-1162}
}
```

Citing contexts in the V6 manuscript:
- ...callum1998comparison}, Logistic Regression models, \cite{genkin01082007}, and Support Vector Machines \cite{joachims1998text}. The emergence of deep neural architectures and more powerful processors designed specifically for ML and AI type computations facilitated the advance of word embedding language models such as Word2Vec \cite{mikolov}, GloVe \cite{pennington}, and FastText \cite{articleboja} which were then used in DL models to push the boundaries for many NLP tasks, text classification being no exception ...
- ...For static, word-based embeddings, we evaluate GloVe, Word2Vec, and FastText models. These embeddings were trained on the Common Crawl dataset, a large-scale, multilingual web corpus consisting of over 600 billion tokens.\footnote{Source: https://commoncrawl.org/} The GloVe model \cite{pennington} we use is glove.840B.300d.txt which captures global co-occurrence statistics across a large corpus, providing 300-dimensional embeddings. The Word2Ve...
- ...\textbf{Static Language Model Classifiers (Static LM):} The second module employs neural network-based classifiers, originally presented in \cite{moreo2021}, which are designed to leverage pre-trained static word embedding language models such as Word2Vec \cite{mikolov}, GloVe \cite{pennington} and FastText \cite{articleboja} models. We include three different types of neural classifiers in this module, each of which is constructed using Pyt...
- ...or the model.} using either a continuous bag of words (CBOW) approach where the model is optimized to predict the target word given a specific context, or a Skip-Gram predictive approach where the context of a word is predicted given the target word. GloVe (Global Vectors for Word Representation), another form of word embeddings introduced in 2014 \cite{pennington}, leverages both local and global word context to generate word embeddings, explicitly constructing a word co-occurrence matrix during training that r...

## moreo2021

```bibtex
@article{moreo2021,
	author = {Moreo, Alejandro and Esuli, Andrea and Sebastiani, Fabrizio},
	date = {2021/05/01},
	doi = {10.1007/s10618-020-00735-3},
	id = {Moreo2021},
	isbn = {1573-756X},
	journal = {Data Mining and Knowledge Discovery},
	number = {3},
	pages = {911--963},
	title = {Word-class embeddings for multiclass text classification},
	url = {https://doi.org/10.1007/s10618-020-00735-3},
	volume = {35},
	year = {2021},
	bdsk-url-1 = {https://doi.org/10.1007/s10618-020-00735-3}
}
```

Citing contexts in the V6 manuscript:
- ...on the requisite computational and system requirements of each of the approaches. The platform that we built to support this testing, which also includes analytics capabilities, we call \textit{Layer Cake}, and it is based off of research and code produced in 2021 which looked at the effect of Word-Class Embeddings (WCEs) on classifier performance \cite{moreo2021}. In this work however, we look at the problem not just from a classifier perspective, but also from a language model and \textit{language representat...
- ...Layer Cake supports a wide array of language models, enabling a comparison of different forms of language representation and an analysis of how model design and training objectives affect classifier effectiveness. While Layer Cake supports various evaluation measures, we primarily focus on Macro-F1 and Micro-F1 scores, following \cite{moreo2021}, as they provide holistic views of classifier performance. ...
- ...\textbf{Static Language Model Classifiers (Static LM):} The second module employs neural network-based classifiers, originally presented in \cite{moreo2021}, which are designed to leverage pre-trained static word embedding language models such as Word2Vec \cite{mikolov}, GloVe \cite{pennington} and FastTe...
- ... neural network classifier by simply concatenating one embedding form with another, a technique that we test extensively within the context of our ML Classifier module where we test the efficacy of a variety of linear algebraic combinations of various forms of text vectorization, word and token embeddings, in conjunction with Word-Class embeddings \cite{moreo2021}, as well as feature reduction techniques such as LSA and document encoding using various embedding pooling strategies....
- ...te{zhang2015character}, Long Short-Term Memory networks (LSTM) \cite{zang2021research}, and Attention based neural networks (ATTN) \cite{liuguo2019bidirectional}, each of which is used as benchmark classifiers with Layer Cake to test different static word embedding language models (Word2Vec, GloVe and Fasttext) in a deep learning setting following \cite{moreo2021}. Each of these neural models is tested with each of the static embedding language models using a variety of hyperparameter settings and generally spe...
- ...the number of classes or labels in the text classification problem, that maps each specific term or word of the language model dictionary to each of the predefined class labels, calculated using a variety of techniques that range from the straight probabilistic to the use of informational theoretical primitives such as pointwise mutual information \cite{moreo2021}. Confining the WCEs to their own vector space allows for them to be added to pretrained (static) word embeddings quite elegantly, and their work show...
- ...problem in question, determines the forms of language representation that can be used for that given classifier as well as the data preprocessing steps necessary to configure such language representation forms. The overall dataflow and modular architecture of Layer Cake is depicted in Figure \ref{fig6}. While our test methodology generally follows \cite{moreo2021}, we extend the footprint significantly in terms of the ML classifiers and language models we test, the forms of language representation that we use f...
- ...The neural model classifiers that are designed for the static word embedding language models - Word2Vec, fastText and GloVe - are GPU bound (CUDA), although they also run on MPS / Apple Silicon. The neural classifiers used for this testing are hand crafted using a very thin PyTorch layered design which is inherited from \cite{moreo2021} that supports custom designed CNN, ATTN and LSTM based neural classifiers. These classifiers do not support any form of parallelism however, and ther...
- ...nsformers) into the Hugging Face classifiers designed to support transformer-based language models. This was in fact the original motivation for this work was to test the efficacy of integrating varying types of embeddings that were generated outside of the model within the context of text classification problem, work which is the very crux of the \cite{moreo2021} study. Given the constraints of the underlying Hugging Face transformer models however, as well as the tight coupling between the initial embedding l...
- ...t we tried, regardless of where in the neural model we tried it, these TCEs that are constructed in their own latent vector space and have shown to be effective in text classification problems in both classical ML classifiers andDL classifiers that are designed to support static language model embeddings, as we report here which is consistent with \cite{moreo2021}, not only did their integration \textit{not} improve the classifier model performance (in terms of Macro-F1 and Micro-F1 scores), but it actually \te...
- ...The train, validation and test split numbers for each of the datasets is listed in the table below, and is broken down by classifier as each classifier has slightly different preprocessing logic, with the neural networks requiring a validation data set set aside from the training data to test the training epochs. Note that we, following \cite{moreo2021}, use the standard 'Lewis Split' for the RCV-1 dataset \cite{lewis2004}, allowing us to evaluate our results more precisely against prior research....
- ...er insight that is revealed within the context of this research is that with the static embedding (language) models, one has a great deal of flexibility with respect to how the embedding layer, which is the storehouse of the word-embedding matrix basically, is setup and initialized. The fundamental method that we use for our tests, again following \cite{moreo2021}, is to set up the embedding layer with only the necessary and requisite embeddings in the dataset that we are solving for, effectively reducing the s...
- ...Each of these representation form types is tested with and without Word-Class Embeddings\footnote{As defined in the study of the same name published in 2021 \cite{moreo2021} which forms the basis of the foundational code for Layer Cake.}, given that we have the ability to perform various forms of linear algebraic transfor...
- ...(LSA), sometimes called Latent Semantic Indexing (LSI), \cite{deerwester} into the ML representation form mix as well, a method that continues to show its value as an extremely effective model optimization technique. We even show that LSA can be used in conjunction with WCEs / TCEs to boost model effectiveness, consistent with the results found by \cite{moreo2021} with respect to WCEs boosting the performance of DL approaches with static embeddings. LSA uses a matrix factorization technique called singular valu...
- ...at are included in the output tf-idf vectors to those tokens that are found in at least 5 documents, a form of feature reduction that is sometimes called \textit{cutoff} in the literature, and b) we use the sublinear argument to log normalize the tf-idf values, converting the term frequency (tf) value to 1 + log(tf).\footnote{This approach follows \cite{moreo2021}.} The other representation forms include the embedding\_vocab\_matrix structure which is a static representation of the dictionary of words or tokens...
- ...The first neural model test script is inherited from \cite{moreo2021} but extended to support Word2Vec and FastText in addition to GloVe. The script, the dataset preparation, and the classifier design in this module are...
- ... of the underlying language model we are working with, we then test each of the static word embedding neural classifiers under different hyperparameter settings and with each of the in scope datasets, both with and without WCEs, resulting in metrics - like the initial set of ML classifier data - that can be directed compared against the 2021 study \cite{moreo2021}. During training, these indexed, numerical representations of the documents are batched and passed to the model's embedding layer through the NeuralC...
- ... albeit somewhat less effective than the transformer models. However, one of our unique findings with this research is that despite their performance, these transformer models - at least the ones fromHugging Face - do not have the ability to integrate with externally computed embeddings (i.e. what we call TCEs as a variant of the WCEs evaluated in \cite{moreo2021}....

## schuster

```bibtex
@inproceedings{schuster,
author = {Schuster, Mike and Nakajima, Kaisuke},
year = {2012},
month = {03},
pages = {5149-5152},
title = {Japanese and Korean voice search},
isbn = {978-1-4673-0045-2},
journal = {Acoustics, Speech, and Signal Processing, 1988. ICASSP-88., 1988 International Conference on},
doi = {10.1109/ICASSP.2012.6289079}
}
```

Citing contexts in the V6 manuscript:
- ...Piece, Byte-Pair Encoding, and SentencePiece tokenization, each of which is used by one or more language models that is tested with Layer Cake and each of which, like FastText's tokenization strategy, is a form of subword tokenization. WordPiece is a subword tokenization algorithm originally developed for Google's neural machine translation system \cite{schuster} \cite{wu}, and then used with BERT \cite{devlin-etal-2019-bert}. This tokenization strategy begins with a base vocabulary of individual characters an...

## wu

```bibtex
@article{wu,
author = {Wu, Yonghui and Schuster, Mike and Chen, Zhifeng and Le, Quoc and Norouzi, Mohammad and Macherey, Wolfgang and Krikun, Maxim and Cao, Yuan and Gao, Qin and Macherey, Klaus and Klingner, Jeff and Shah, Apurva and Johnson, Melvin and Liu, Xiaobing and Kaiser, ukasz and Gouws, Stephan and Kato, Yoshikiyo and Kudo, Taku and Kazawa, Hideto and Dean, Jeffrey},
year = {2016},
month = {09},
pages = {},
title = {Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation},
doi = {10.48550/arXiv.1609.08144}
}
```

Citing contexts in the V6 manuscript:
- ... Encoding, and SentencePiece tokenization, each of which is used by one or more language models that is tested with Layer Cake and each of which, like FastText's tokenization strategy, is a form of subword tokenization. WordPiece is a subword tokenization algorithm originally developed for Google's neural machine translation system \cite{schuster} \cite{wu}, and then used with BERT \cite{devlin-etal-2019-bert}. This tokenization strategy begins with a base vocabulary of individual characters and then ite...

## devlin-etal-2019-bert

```bibtex
@inproceedings{devlin-etal-2019-bert,
    title = "{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    author = "Devlin, Jacob  and
      Chang, Ming-Wei  and
      Lee, Kenton  and
      Toutanova, Kristina",
    editor = "Burstein, Jill  and
      Doran, Christy  and
      Solorio, Thamar",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-1423/",
    doi = "10.18653/v1/N19-1423",
    pages = "4171--4186",
   
}
```

Citing contexts in the V6 manuscript:
- ...ethods of training include \textit{next token prediction}\footnote{Next token prediction based LLMs are called \textit{causal} language models due to the determinative nature of the underlying algorithm.} which is used for example by the GPT family of models built by OpenAI \cite{radford2019language}, and \textit{masking} which is used by the BERT \cite{devlin-etal-2019-bert} and RoBERTa \cite{liu2019roberta} based language model families models. All of these transformer-based LLMs however ultimately leverage these static,...
- ...t \cite{articleboja} which were then used in DL models to push the boundaries for many NLP tasks, text classification being no exception \cite{zhang2015character} \cite{yoon2014convolutional} \cite{hochreiterlong}. Yet it was the introduction of transformer architecture in 2017 \cite{vaswani} followed shortly thereafter by the introduction of BERT \cite{devlin-etal-2019-bert} which demonstrated how powerful LLMs could be in establishing new benchmarks for NLG and NLP solves. The effectiveness of LLMs in this context demons...
- ...The supported transformer-based language models we support, models we refer to as 'context-aware' given their support for attention masking and positional encoding information, include language models from the BERT, RoBERTa, DistilBERT (distilbert-base-cased), XLNet, GPT-2, Llama, and DeepSeek families (see Table 2 above for details). BERT \cite{devlin-etal-2019-bert} and RoBERTa \cite{liu2019roberta} are pretrained using masked language modeling (MLM), with BERT also using next sentence prediction (NSP). RoBERTa m...
- ...\textbf{Transformer-based Classifiers:} The final module incorporates transformer-based classifiers using the Hugging Face Transformers library\footnote{https://huggingface.co/docs/transformers/en/index}. We support two primary architectures: the standard Sequence Classification model, which utilizes pretrained transformer models such as BERT \cite{devlin-etal-2019-bert}, RoBERTa \cite{liu2019roberta}, DistilBERT \cite{sanh2020distilbert}, XLNet \cite{yang2019xlnet}, GPT-2 \cite{radford2019language}, Llama \cite{touvr...
- ...zation, each of which is used by one or more language models that is tested with Layer Cake and each of which, like FastText's tokenization strategy, is a form of subword tokenization. WordPiece is a subword tokenization algorithm originally developed for Google's neural machine translation system \cite{schuster} \cite{wu}, and then used with BERT \cite{devlin-etal-2019-bert}. This tokenization strategy begins with a base vocabulary of individual characters and then iteratively builds a vocabulary of subwords by merging th...
- ...Furthermore, some language models also include what are known as segment embeddings, which provide information about which sentence, and which part of which sentence, each token belongs to, requisite information for LLM training tasks like next-sentence prediction for example. Models such as BERT \cite{devlin-etal-2019-bert}, ALBERT \cite{lan2019albert}, XLNet \cite{yang2019xlnet}, and ELECTRA \cite{clark2020electra} all leverage some form of segment embeddings, learnable...
- ...xtualized (dynamic) text representation forms of natural language were ELMo (Embeddings from Language Models), which was introduced in 2018 \cite{peters2018deep} which used bidirectional LSTM layers to generate contextualized embeddings, and then of course BERT (Bidirectional Encoder Representations from Transformers), which was introduced in 2019 \cite{devlin-etal-2019-bert} which uses a unique, bidirectional approach to generating contextualized embeddings which, in both its original form and in many branches of derivati...
- ...in use right now leverage this transformer, attention-based architecture which has replaced virtually all of the prior DL, neural network based architectures that came before it (e.g. RNNs with self-attention mechanisms or Convolutional Neural Networks or CNNs), underpinning many state-of-the-art results in machine translation, text classification \cite{devlin-etal-2019-bert}, natural language understanding, and (natural) language generation tools \cite{radford2019language} and applications. These models, and their associa...
- ... for the given language model and is defined by the training algorithm and pre-trained data which the language model was trained on. Some of the earliest transformer models that were shown to be effective for text classification problems were BERT, which stands for Bidirectional Encoder Representations from Transformers which was released in 2019 \cite{devlin-etal-2019-bert}, as well as derivations thereof: like RoBERTa, or Robustly Optimized BERT Pretraining Approach which was released shortly after BERT also in 2019 \ci...

## raffel2019exploring

```bibtex
@article{raffel2019exploring,
  title={Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  author={Raffel, Colin and Shazeer, Noam and Roberts, Adam and Lee, Katherine and Narang, Sharan and Matena, Michael and Zhou, Yanqi and Li, Wei and Liu, Peter J.},
  journal={arXiv preprint arXiv:1910.10683},
  year={2019},
  doi={10.48550/arXiv.1910.10683},
  url={https://doi.org/10.48550/arXiv.1910.10683}
}
```

Citing contexts in the V6 manuscript:
- ...ed for data compression and was then later adapted for NLP by the SentencePiece and GPT/GPT-2 models. Byte Pair Encoding begins with characters as tokens and then repeatedly merges the most frequent pair of tokens (characters or subwords) into a new tokens, continuing until reaching a desired vocabulary size. SentencePiece tokenization, used by T5 \cite{raffel2019exploring}, Albert \cite{lan2019albert} and XLNet \cite{yang2019xlnet} models, is a language-independent, unsupervised tokenization method introduced by Google ...

## lan2019albert

```bibtex
@article{lan2019albert,
  title={ALBERT: A Lite BERT for Self-supervised Learning of Language Representations},
  author={Lan, Zhenzhong and Chen, Mingda and Goodman, Sebastian and Gimpel, Kevin and Sharma, Piyush and Soricut, Radu},
  journal={arXiv preprint arXiv:1909.11942},
  year={2019},
  doi={10.48550/arXiv.1909.11942},
  url={https://doi.org/10.48550/arXiv.1909.11942}
}
```

Citing contexts in the V6 manuscript:
- ...n later adapted for NLP by the SentencePiece and GPT/GPT-2 models. Byte Pair Encoding begins with characters as tokens and then repeatedly merges the most frequent pair of tokens (characters or subwords) into a new tokens, continuing until reaching a desired vocabulary size. SentencePiece tokenization, used by T5 \cite{raffel2019exploring}, Albert \cite{lan2019albert} and XLNet \cite{yang2019xlnet} models, is a language-independent, unsupervised tokenization method introduced by Google in 2018 for neural text proce...
- ...Furthermore, some language models also include what are known as segment embeddings, which provide information about which sentence, and which part of which sentence, each token belongs to, requisite information for LLM training tasks like next-sentence prediction for example. Models such as BERT \cite{devlin-etal-2019-bert}, ALBERT \cite{lan2019albert}, XLNet \cite{yang2019xlnet}, and ELECTRA \cite{clark2020electra} all leverage some form of segment embeddings, learnable vectors that differentiate t...

## yang2019xlnet

```bibtex
@inproceedings{yang2019xlnet,
  title={XLNet: Generalized Autoregressive Pretraining for Language Understanding},
  author={Yang, Zhilin and Dai, Zihang and Yang, Yiming and Carbonell, Jaime and Salakhutdinov, Ruslan and Le, Quoc V.},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  pages={5753--5763},
  year={2019},
  publisher={Curran Associates, Inc.},
  address={Red Hook, NY}
}
```

Citing contexts in the V6 manuscript:
- ..., with BERT also using next sentence prediction (NSP). RoBERTa modifies BERT by removing NSP and adjusting pretraining strategies to improve generalization. DistilBERT \cite{sanh2020distilbert}, based on BERT, uses knowledge distillation \cite{hinton2015distilling} to create a lighter, faster model while retaining most of BERT's performance. XLNet \cite{yang2019xlnet} builds on Transformer-XL \cite{dai2019transformer} and introduces permutation-based training to capture bidirectional context without masking, blendi...
- ...using the Hugging Face Transformers library\footnote{https://huggingface.co/docs/transformers/en/index}. We support two primary architectures: the standard Sequence Classification model, which utilizes pretrained transformer models such as BERT \cite{devlin-etal-2019-bert}, RoBERTa \cite{liu2019roberta}, DistilBERT \cite{sanh2020distilbert}, XLNet \cite{yang2019xlnet}, GPT-2 \cite{radford2019language}, Llama \cite{touvron2023llama}, and DeepSeek \cite{guo2025deepseek}; and a transformer-based CNN architecture, inte...
- ...SentencePiece and GPT/GPT-2 models. Byte Pair Encoding begins with characters as tokens and then repeatedly merges the most frequent pair of tokens (characters or subwords) into a new tokens, continuing until reaching a desired vocabulary size. SentencePiece tokenization, used by T5 \cite{raffel2019exploring}, Albert \cite{lan2019albert} and XLNet \cite{yang2019xlnet} models, is a language-independent, unsupervised tokenization method introduced by Google in 2018 for neural text processing \cite{kudo2018sentencepie...
- ...some language models also include what are known as segment embeddings, which provide information about which sentence, and which part of which sentence, each token belongs to, requisite information for LLM training tasks like next-sentence prediction for example. Models such as BERT \cite{devlin-etal-2019-bert}, ALBERT \cite{lan2019albert}, XLNet \cite{yang2019xlnet}, and ELECTRA \cite{clark2020electra} all leverage some form of segment embeddings, learnable vectors that differentiate tokens belonging to different...
- ...These transformer models, each with its own nuanced training approach and transformer architecture, along with the GPT-3 model \cite{articlebrown}, as well as an XLNet model \cite{yang2019xlnet}, along with smaller language models from both the Llama \cite{touvron2023llama} and DeepSeek family \cite{articlebi}, were included in the benchmarki...

## kudo2018sentencepiece

```bibtex
@article{kudo2018sentencepiece,
  title={SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing},
  author={Kudo, Taku and Richardson, John},
  journal={arXiv preprint arXiv:1808.06226},
  year={2018},
  month={Aug},
  doi={10.48550/arXiv.1808.06226},
  url={https://doi.org/10.48550/arXiv.1808.06226}
}
```

Citing contexts in the V6 manuscript:
- ...ns (characters or subwords) into a new tokens, continuing until reaching a desired vocabulary size. SentencePiece tokenization, used by T5 \cite{raffel2019exploring}, Albert \cite{lan2019albert} and XLNet \cite{yang2019xlnet} models, is a language-independent, unsupervised tokenization method introduced by Google in 2018 for neural text processing \cite{kudo2018sentencepiece}. Unlike traditional tokenizers, it does not rely on whitespace for tokenization, which is important when supporting languages that do not have clear ...

## clark2020electra

```bibtex
@article{clark2020electra,
  title={ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators},
  author={Clark, Kevin and Luong, Minh-Thang and Le, Quoc V. and Manning, Christopher D.},
  journal={arXiv preprint arXiv:2003.10555},
  year={2020},
  doi={10.48550/arXiv.2003.10555},
  url={https://doi.org/10.48550/arXiv.2003.10555}
}
```

Citing contexts in the V6 manuscript:
- ...what are known as segment embeddings, which provide information about which sentence, and which part of which sentence, each token belongs to, requisite information for LLM training tasks like next-sentence prediction for example. Models such as BERT \cite{devlin-etal-2019-bert}, ALBERT \cite{lan2019albert}, XLNet \cite{yang2019xlnet}, and ELECTRA \cite{clark2020electra} all leverage some form of segment embeddings, learnable vectors that differentiate tokens belonging to different input sentences within a single inpu...

